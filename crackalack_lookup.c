/*
 * Rainbow Crackalack: crackalack_lookup.c
 * Copyright (C) 2018-2021  Joe Testa <jtesta@positronsecurity.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms version 3 of the GNU General Public License as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * Performs GPU-accelerated password hash lookups on rainbow tables.
 */

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/sysinfo.h>
#define O_BINARY 0
#endif

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <locale.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "opencl_setup.h"

#include "charset.h"
#include "clock.h"
#include "cpu_rt_functions.h"
#include "hash_validate.h"
#include "misc.h"
#include "ntlmv1.h"
#include "rtc_decompress.h"
#include "shared.h"
#include "test_shared.h"  /* TODO: move hex_to_bytes() elsewhere. */
#include "verify.h"
#include "version.h"

#define VERBOSE 1
#define PRECOMPUTE_KERNEL_PATH "precompute.cl"
#define PRECOMPUTE_NTLM8_KERNEL_PATH "precompute_ntlm8.cl"
#define PRECOMPUTE_NTLM9_KERNEL_PATH "precompute_ntlm9.cl"
#define PRECOMPUTE_NETNTLMV1_KERNEL_PATH "precompute_netntlmv1.cl"

#define FALSE_ALARM_KERNEL_PATH "false_alarm_check.cl"
#define FALSE_ALARM_NTLM8_KERNEL_PATH "false_alarm_check_ntlm8.cl"
#define FALSE_ALARM_NTLM9_KERNEL_PATH "false_alarm_check_ntlm9.cl"
#define FALSE_ALARM_NETNTLMV1_KERNEL_PATH "false_alarm_check_netntlv1.cl"

/* Number of tables whose binary-search results are accumulated per worker
 * before triggering one GPU FA launch.  Increases FA kernel occupancy from
 * ~0.5% per table to ~2% at 4, reducing total FA wall-clock time. */
#define FA_BATCH_SIZE 4

#define HASH_FILE_FORMAT_PLAIN 1
#define HASH_FILE_FORMAT_PWDUMP 2


/* Struct to form a linked list of precomputed end indices, and potential start indices (which are usually false alarms). */
struct _precomputed_and_potential_indices {
  char *username;  /* Non-NULL if loaded file format is pwdump. */
  char *hash;
  cl_ulong *precomputed_end_indices;
  cl_uint num_precomputed_end_indices;

  cl_ulong *potential_start_indices;
  unsigned int num_potential_start_indices;
  unsigned int potential_start_indices_size;
  unsigned int *potential_start_index_positions; /* Buffer size is always num_potential_start_indices. */

  char *plaintext;        /* Set if hash is cracked. */
  unsigned char cracked_key[8];   /* raw recovered plaintext bytes (netntlmv1: 7 key bytes) */
  unsigned int  cracked_key_len;  /* 0 until cracked */
  char *index_filename;   /* File path containing the ".index" file. */
  struct _precomputed_and_potential_indices *next;
};
typedef struct _precomputed_and_potential_indices precomputed_and_potential_indices;


/* Struct to represent one GPU device. */
typedef struct {
  cl_uint device_number;
  cl_device_id device;
  cl_context context;
  cl_program program;
  cl_kernel kernel;
  cl_command_queue queue;
  cl_uint num_work_units;
  size_t tuned_gws;
  /* Cached false alarm kernel — reused across tables to avoid repeated JIT
   * compilation.  Populated on first use and written back by table_worker_thread. */
  cl_context fa_context;
  cl_command_queue fa_queue;
  cl_program fa_program;
  cl_kernel fa_kernel;
} gpu_dev;


/* Struct to pass arguments to a host thread. */
typedef struct {
  unsigned int hash_type;
  char *hash_name;
  char *username; /* Non-NULL when pwdump format input file given. */
  char *hash; /* In hex. */
  char *charset;
  char *charset_name;
  unsigned int plaintext_len_min;
  unsigned int plaintext_len_max;
  unsigned int table_index;
  unsigned int reduction_offset;
  unsigned int chain_len;

  unsigned int total_devices;
  uint64_t *results;
  unsigned int num_results;

  cl_ulong *potential_start_indices;
  unsigned int num_potential_start_indices;
  
  /* Buffer size is always num_potential_start_indices. */
  unsigned int *potential_start_index_positions;
  
  /* Length is always num_potential_start_indices. */
  cl_ulong *hash_base_indices;

  gpu_dev gpu;
  int use_barrier;
} thread_args;


/* Per-ppi-node scratch buffer owned by one worker during a single table lookup. */
typedef struct {
  cl_ulong *potential_start_indices;
  unsigned int *potential_start_index_positions;
  unsigned int num_potential_start_indices;
  unsigned int potential_start_indices_size;
} ppi_scratch_entry;

/* Per-worker scratch: one entry per ppi node, plus a mutex for intra-worker concurrency. */
typedef struct {
  ppi_scratch_entry *entries;
  unsigned int num_entries;
  pthread_mutex_t lock;
} worker_scratch;

/* Struct to pass to binary search threads. */
typedef struct {
  cl_ulong *rainbow_table;
  unsigned int num_chains;
  precomputed_and_potential_indices *ppi_head;
  unsigned int thread_number;
  unsigned int total_threads;
  worker_scratch *scratch;
} search_thread_args;


/* Struct to hold node in linked list of preloaded tables. */
struct _preloaded_table {
  char *filepath;
  cl_ulong *rainbow_table;
  unsigned int num_chains;
  struct _preloaded_table *next;
};
typedef struct _preloaded_table preloaded_table;

typedef struct {
  char *rt_dir;
} preloading_thread_args;


/* Args for one worker in the parallel table-search pool. */
typedef struct {
  precomputed_and_potential_indices *ppi_head;
  thread_args *all_device_args;
  unsigned int num_devices;
  unsigned int bs_threads;
  unsigned int worker_id;
  unsigned int total_tables;
} worker_thread_args;


unsigned int count_tables(char *dir);
void find_rt_params(char *dir, rt_parameters *rt_params);
void free_loaded_hashes(char **usernames, char **hashes);
void *host_thread_false_alarm(void *ptr);
void *preloading_thread(void *ptr);
void print_eta_precompute();
cl_ulong *search_precompute_cache(char *index_data, unsigned int *num_indices, char *filename, unsigned int filename_size);
void search_tables(unsigned int total_tables, precomputed_and_potential_indices *ppi, thread_args *args);
void save_cracked_hash(precomputed_and_potential_indices *ppi, unsigned int hash_type);
void check_false_alarms_worker(precomputed_and_potential_indices *ppi_head, worker_scratch *scratch, thread_args *fa_args, unsigned int dev_slot);
void *table_worker_thread(void *ptr);
worker_scratch *alloc_worker_scratch(precomputed_and_potential_indices *ppi_head);
void free_worker_scratch(worker_scratch *scratch);
void clear_worker_scratch(worker_scratch *scratch);


/* The path of the pot file to store cracked hashes in.  This can be overridden by
 * a command line arg. */
char jtr_pot_filename[128] = "rainbowcrackalack_jtr.pot";
char hashcat_pot_filename[128] = "rainbowcrackalack_hashcat.pot";

/* The number of seconds spent on precomputation, file I/O, searching, and false alarm
 * checking. */
double time_precomp = 0, time_io = 0, time_searching = 0, time_falsealarms = 0;

/* The total number of false alarms, chains processed, respectively. */
uint64_t num_falsealarms = 0, num_chains_processed = 0;

/* The total number of hashes cracked in this invokation and number of tables
 * processed, respectively. */
unsigned int num_cracked = 0, num_tables_processed = 0;

/* Mutex to protect the precomputed_and_potential_indices array. _*/
pthread_mutex_t ppi_mutex = PTHREAD_MUTEX_INITIALIZER;

/* Barrier to ensure that kernels on multiple devices are all run at the same time.
 * The closed-source AMD driver on Windows effectively blocks other devices while
 * one kernel is running; this ensures parallelization in that environment, since
 * all kernels will run at once.  The open source AMD ROCm driver on Linux may or
 * may not get a very slight performance bump with this enabled. */
pthread_barrier_t barrier = {0};

/* Set to 1 if AMD GPUs found. */
unsigned int is_amd_gpu = 0;

/* The global work size, as over-ridden by the user on the command line. */
size_t user_provided_gws = 0;

/* The platform number to disable (-1 to not disable any). */
int disable_platform = -1;

/* The total number of precomputed indices loaded into memory.  Each one of these is
 * a cl_ulong (8 bytes). */
uint64_t total_precomputed_indices_loaded = 0;

/* Set to 1 if the NTLM8/9 message was printed.  This prevents console spam. */
unsigned int printed_precompute_optimized_message = 0;
unsigned int printed_false_alarm_optimized_message = 0;

/* The total number of tables in all subdirectories of the directory given
 * by the user. */
unsigned int total_tables = 0;

/* Set to 1 by the preloading thread to indicate that no more tables exist for loading. */
unsigned int table_loading_complete = 0;

/* The current size of the preloaded tables list. */
unsigned int num_preloaded_tables_available = 0;

/* A linked list of preloaded tables. */
preloaded_table *preloaded_table_list = NULL;

/* Condition for the main thread to wait for more tables on. */
pthread_cond_t condition_wait_for_tables = PTHREAD_COND_INITIALIZER;

/* Condition for the preloading thread to wait on (when the MAX_PRELOAD_NUM is reached). */
pthread_cond_t condition_continue_loading_tables = PTHREAD_COND_INITIALIZER;

/* The lock for the preloaded tables system. */
pthread_mutex_t preloaded_tables_lock = PTHREAD_MUTEX_INITIALIZER;

/* The time at which precomputation begins. */
struct timespec precompute_start_time = {0};

/* The time at which table searching begins. */
struct timespec search_start_time = {0};

/* Timestamp of the last ETA progress line; reset at search start. */
struct timespec last_eta_print = {0};

/* Number of uncracked hashes. */
unsigned int num_hashes = 0;

/* Number of hashes precomputed so far. */
unsigned int num_hashes_precomputed = 0;

/* Total number of hashes that will be precomputed. */
unsigned int num_hashes_precomputed_total = 0;


/* The total number of tables to preload in memory; set to W+1 in search_tables(). */
unsigned int max_preload_num = 2;

#define LOCK_PPI() \
  if (pthread_mutex_lock(&ppi_mutex)) { perror("Failed to lock mutex"); exit(-1); }

#define UNLOCK_PPI() \
  if (pthread_mutex_unlock(&ppi_mutex)) { perror("Failed to unlock mutex"); exit(-1); }


pthread_mutex_t stats_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t potfile_mutex = PTHREAD_MUTEX_INITIALIZER;

/* Device pool: stack of available device indices for worker false-alarm checks. */
pthread_mutex_t device_pool_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t device_pool_cond = PTHREAD_COND_INITIALIZER;
unsigned int device_pool_slots[MAX_NUM_DEVICES];
unsigned int device_pool_count = 0;

unsigned int all_cracked = 0;
unsigned int stop_preloading = 0;

int ntlmv1_mode = 0;
char *ntlmv1_capture_arg = NULL;
int ntlmv1_batch_mode = 0;
char *ntlmv1_file_arg = NULL;

/* Running count of tables successfully added to the preload buffer; used for
 * diagnostic logging only. */
unsigned int tables_preloaded_count = 0;


worker_scratch *alloc_worker_scratch(precomputed_and_potential_indices *ppi_head) {
  worker_scratch *scratch = NULL;
  precomputed_and_potential_indices *ppi_cur = ppi_head;
  unsigned int count = 0;

  while (ppi_cur != NULL) { count++; ppi_cur = ppi_cur->next; }

  scratch = calloc(1, sizeof(worker_scratch));
  if (scratch == NULL) { fprintf(stderr, "Failed to alloc worker_scratch.\n"); exit(-1); }

  scratch->entries = calloc(count, sizeof(ppi_scratch_entry));
  if (scratch->entries == NULL) { fprintf(stderr, "Failed to alloc scratch entries.\n"); exit(-1); }

  scratch->num_entries = count;
  pthread_mutex_init(&scratch->lock, NULL);
  return scratch;
}


void free_worker_scratch(worker_scratch *scratch) {
  unsigned int i = 0;
  for (i = 0; i < scratch->num_entries; i++) {
    FREE(scratch->entries[i].potential_start_indices);
    FREE(scratch->entries[i].potential_start_index_positions);
  }
  FREE(scratch->entries);
  pthread_mutex_destroy(&scratch->lock);
  FREE(scratch);
}


void clear_worker_scratch(worker_scratch *scratch) {
  unsigned int i = 0;
  for (i = 0; i < scratch->num_entries; i++) {
    FREE(scratch->entries[i].potential_start_indices);
    FREE(scratch->entries[i].potential_start_index_positions);
    scratch->entries[i].num_potential_start_indices = 0;
    scratch->entries[i].potential_start_indices_size = 0;
  }
}


void scratch_add_match(worker_scratch *scratch, unsigned int ppi_idx, cl_ulong start, unsigned int position) {
  #define SCRATCH_INITIAL_SIZE 16
  ppi_scratch_entry *e = NULL;

  pthread_mutex_lock(&scratch->lock);
  e = &scratch->entries[ppi_idx];

  if (e->potential_start_indices == NULL) {
    e->potential_start_indices = calloc(SCRATCH_INITIAL_SIZE, sizeof(cl_ulong));
    e->potential_start_index_positions = calloc(SCRATCH_INITIAL_SIZE, sizeof(unsigned int));
    if ((e->potential_start_indices == NULL) || (e->potential_start_index_positions == NULL)) {
      fprintf(stderr, "Failed to init scratch entry buffers.\n"); exit(-1);
    }
    e->potential_start_indices_size = SCRATCH_INITIAL_SIZE;
  }

  if (e->num_potential_start_indices == e->potential_start_indices_size) {
    unsigned int new_size = e->potential_start_indices_size * 2;
    e->potential_start_indices = recalloc(e->potential_start_indices, new_size * sizeof(cl_ulong), e->potential_start_indices_size * sizeof(cl_ulong));
    e->potential_start_index_positions = recalloc(e->potential_start_index_positions, new_size * sizeof(unsigned int), e->potential_start_indices_size * sizeof(unsigned int));
    if ((e->potential_start_indices == NULL) || (e->potential_start_index_positions == NULL)) {
      fprintf(stderr, "Failed to resize scratch entry buffers.\n"); exit(-1);
    }
    e->potential_start_indices_size = new_size;
  }

  e->potential_start_indices[e->num_potential_start_indices] = start;
  e->potential_start_index_positions[e->num_potential_start_indices] = position;
  e->num_potential_start_indices++;

  pthread_mutex_unlock(&scratch->lock);
}


void check_false_alarms(precomputed_and_potential_indices *ppi, thread_args *args) {
  pthread_t threads[MAX_NUM_DEVICES] = {0};
  char time_str[128] = {0};
  struct timespec start_time = {0};
  cl_ulong plaintext_space_up_to_index[MAX_PLAINTEXT_LEN] = {0};

  unsigned int num_potential_start_indices = 0, i = 0, j = 0; // init to -1 since 0 is possible index
  unsigned int total_devices = args[0].total_devices;
  cl_ulong plaintext_space_total = 0;
  double time_delta = 0.0;

  precomputed_and_potential_indices *ppi_cur = ppi;
  cl_ulong *potential_start_indices = NULL, *hash_base_indices = NULL;
  unsigned int *potential_start_index_positions = NULL;
  precomputed_and_potential_indices **ppi_refs = NULL;


  /* First count all the potential start indices. */
  while(ppi_cur) {
    num_potential_start_indices += ppi_cur->num_potential_start_indices;
    ppi_cur = ppi_cur->next;
  }
  // nic come back
  /* If no potential matches were found, there's nothing else to do. */
  if (num_potential_start_indices == 0) { // was 0
    printf("No matches found in table.\n");
    return;
  }
  printf("  Checking %u potential matches...\n", num_potential_start_indices);  fflush(stdout);
  num_falsealarms += num_potential_start_indices;

  /* Allocate a buffer to hold them all. */
  potential_start_indices = calloc(num_potential_start_indices, sizeof(cl_ulong));
  potential_start_index_positions = calloc(num_potential_start_indices, sizeof(cl_ulong));
  hash_base_indices = calloc(num_potential_start_indices, sizeof(cl_ulong));
  ppi_refs = calloc(num_potential_start_indices, sizeof(precomputed_and_potential_indices *));
  if ((potential_start_indices == NULL) || (potential_start_index_positions == NULL) || (hash_base_indices == NULL) || (ppi_refs == NULL)) {
    fprintf(stderr, "Error while creating buffer for potential start indices/positions/hash indices/ppi refs.\n");
    exit(-1);
  }
  int charset_len = 0;
  if (strcmp(args->charset_name, "byte") == 0) {
    charset_len = 256;
  }
  else {
    charset_len = strlen(args->charset);
  }

  plaintext_space_total = fill_plaintext_space_table(charset_len, args->plaintext_len_min, args->plaintext_len_max, plaintext_space_up_to_index);

  /* Collate all the start indices into one buffer. */
  ppi_cur = ppi;
  while(ppi_cur) {
    unsigned char hash[MAX_HASH_OUTPUT_LEN] = {0};
    unsigned int hash_len = hex_to_bytes(ppi_cur->hash, sizeof(hash), hash);
    cl_ulong hash_base_index = hash_to_index(hash, hash_len, args->reduction_offset, plaintext_space_total, 0);  /* We always use position 0 here.  When the GPU code is comparing indices, it will add in the current position. */


    if (ppi_cur->plaintext == NULL) {
      for (i = 0; i < ppi_cur->num_potential_start_indices; i++, j++) {
	potential_start_indices[j] = ppi_cur->potential_start_indices[i];
	potential_start_index_positions[j] = ppi_cur->potential_start_index_positions[i];
	hash_base_indices[j] = hash_base_index;

	/* For this index, hold a reference to the ppi struct.  This later lets us find
	 * the ppi, given a result index from the GPU. */
	ppi_refs[j] = ppi_cur;
      }
    }

    ppi_cur = ppi_cur->next;
  }

  /*for (i = 0; i < num_potential_start_indices; i++)
    printf("Start point: %lu; Chain position: %u; hash base index: %lu\n", potential_start_indices[i], potential_start_index_positions[i], hash_base_indices[i]);*/

  /* Start the timer false alarm checking. */
  start_timer(&start_time);

  /* Start one thread to control each GPU. */
  for (i = 0; i < total_devices; i++) {

    /* Each thread gets the same reference to the list of potential start indices. */
    args[i].potential_start_indices = potential_start_indices;
    args[i].num_potential_start_indices = num_potential_start_indices;
    args[i].potential_start_index_positions = potential_start_index_positions;
    args[i].hash_base_indices = hash_base_indices;

    if (pthread_create(&(threads[i]), NULL, &host_thread_false_alarm, &(args[i]))) {
      perror("Failed to create thread");
      exit(-1);
    }
    //printf("********************************** host_thread_false_alarm created\n");
  }

  /* Wait for all threads to finish. */
  for (i = 0; i < total_devices; i++) {
    if (pthread_join(threads[i], NULL) != 0) {
      perror("Failed to join with thread");
      exit(-1);
    }
  }
  //printf("********************************** host_thread_false_alarm joined\n");

  /* Search for valid results, and update the ppi with the plaintext. */
  for (i = 0; i < total_devices; i++) {
    for (j = 0; j < args[i].num_results; j++) {
      if (args[i].results[j] != 0) {
      	char plaintext[MAX_PLAINTEXT_LEN] = {0};
      	unsigned int plaintext_len = 0;


      	index_to_plaintext(args[i].results[j], args[i].charset, charset_len, args[i].plaintext_len_min, args[i].plaintext_len_max, plaintext_space_up_to_index, plaintext, &plaintext_len);

      	/* Double check NTLM results to weed out super false alarms. */
      	if (args[i].hash_type == HASH_NTLM) {
      	  unsigned char hash[16] = {0};
      	  char hash_hex[(sizeof(hash) * 2) + 1] = {0};


      	  ntlm_hash(plaintext, plaintext_len, hash);
      	  if (!bytes_to_hex(hash, sizeof(hash), hash_hex, sizeof(hash_hex)) || \
      	      (strcmp(hash_hex, ppi_refs[j]->hash) != 0)) {
      	    /*printf("Found super false positive!: NTLM('%s') != %s\n", plaintext, ppi_refs[j]->hash);*/
      	    continue;
      	  }
      	} else if (args[i].hash_type == HASH_NETNTLMV1) {

          unsigned char hash[8] = {0};
          char hash_hex[(sizeof(hash) * 2) + 1] = {0};

          netntlmv1_hash((unsigned char *)plaintext, plaintext_len, hash);

          if (!bytes_to_hex(hash, sizeof(hash), hash_hex, sizeof(hash_hex)) || \
              (strncmp(hash_hex, ppi_refs[j]->hash, 16) != 0)) {
                printf("Found super false positive!: (Net-NTLMv1('%s') == %s) != %s\n", plaintext, hash_hex, ppi_refs[j]->hash);
            continue;
          }
        } else {
      	  printf("WARNING: CPU code to double-check this cracked hash has not yet been added.  There is a 60%% chance this is a false positive!  A workaround is to use John The Ripper to validate this result(s).\n");
        }

      	/* Its official: we cracked a hash! */

      	/* Save the plaintext, clear the precomputed end indices list (since its
      	 * no longer useful, save the hash/plaintext combo into the pot file, and
      	 * tell the user. */
      	ppi_refs[j]->plaintext = strdup(plaintext);
      	ppi_refs[j]->cracked_key_len = (plaintext_len < (unsigned int)sizeof(ppi_refs[j]->cracked_key))
      	    ? plaintext_len : (unsigned int)sizeof(ppi_refs[j]->cracked_key);
      	memcpy(ppi_refs[j]->cracked_key, plaintext, ppi_refs[j]->cracked_key_len);
      	ppi_refs[j]->num_precomputed_end_indices = 0;
      	FREE(ppi_refs[j]->precomputed_end_indices);

      	save_cracked_hash(ppi_refs[j], args[i].hash_type);
        if (args[i].hash_type == HASH_NETNTLMV1) {
          char ptxt_hex[(sizeof(plaintext) * 2) + 1] = {0};
          bytes_to_hex((unsigned char*)plaintext, 7, ptxt_hex, sizeof(ptxt_hex));

          printf("%sHASH CRACKED => %s:1122334455667788:%s%s\n", GREENB, ppi_refs[j]->hash, ptxt_hex, CLR);
          fflush(stdout);
        } else {
          printf("%sHASH CRACKED => %s:1122334455667788:%s%s\n", GREENB, (ppi_refs[j]->username != NULL) ? ppi_refs[j]->username : ppi_refs[j]->hash, plaintext, CLR);  fflush(stdout);
        }
      }
    }
  }
  time_delta = get_elapsed(&start_time);

  time_falsealarms += time_delta;
  seconds_to_human_time(time_str, sizeof(time_str), (unsigned int)time_delta);
  printf("  Completed false alarm checks in %s.\n", time_str);  fflush(stdout);

  FREE(potential_start_indices);
  FREE(potential_start_index_positions);
  FREE(hash_base_indices);
  FREE(ppi_refs);
  FREE(args->results);
  args->num_results = 0;
}


/* Single-device false alarm check for one worker in the parallel pool.
 * Reads match candidates from scratch (not from ppi nodes), runs them on
 * one GPU device, writes cracked plaintexts back to ppi nodes under ppi_mutex,
 * and writes pot-file entries under potfile_mutex. */
void check_false_alarms_worker(precomputed_and_potential_indices *ppi_head, worker_scratch *scratch, thread_args *fa_args, unsigned int dev_slot) {
  pthread_t fa_thread = {0};
  char time_str[128] = {0};
  struct timespec start_time = {0};
  cl_ulong plaintext_space_up_to_index[MAX_PLAINTEXT_LEN] = {0};
  cl_ulong *potential_start_indices = NULL, *hash_base_indices = NULL;
  unsigned int *potential_start_index_positions = NULL;
  precomputed_and_potential_indices **ppi_refs = NULL;
  precomputed_and_potential_indices *ppi_cur = NULL;
  unsigned int num_potential_start_indices = 0, i = 0, j = 0, ppi_idx = 0;
  cl_ulong plaintext_space_total = 0;
  double time_delta = 0.0;
  int charset_len = 0;

  if (strcmp(fa_args->charset_name, "byte") == 0)
    charset_len = 256;
  else
    charset_len = strlen(fa_args->charset);

  plaintext_space_total = fill_plaintext_space_table(charset_len, fa_args->plaintext_len_min, fa_args->plaintext_len_max, plaintext_space_up_to_index);

  /* Count total potential start indices across all scratch entries. */
  for (ppi_idx = 0; ppi_idx < scratch->num_entries; ppi_idx++)
    num_potential_start_indices += scratch->entries[ppi_idx].num_potential_start_indices;

  if (num_potential_start_indices == 0)
    return;

  printf("  Checking %u potential matches...\n", num_potential_start_indices);  fflush(stdout);

  pthread_mutex_lock(&stats_mutex);
  num_falsealarms += num_potential_start_indices;
  pthread_mutex_unlock(&stats_mutex);

  potential_start_indices = calloc(num_potential_start_indices, sizeof(cl_ulong));
  potential_start_index_positions = calloc(num_potential_start_indices, sizeof(unsigned int));
  hash_base_indices = calloc(num_potential_start_indices, sizeof(cl_ulong));
  ppi_refs = calloc(num_potential_start_indices, sizeof(precomputed_and_potential_indices *));
  if ((potential_start_indices == NULL) || (potential_start_index_positions == NULL) || (hash_base_indices == NULL) || (ppi_refs == NULL)) {
    fprintf(stderr, "Error allocating false alarm buffers.\n"); exit(-1);
  }

  /* Flatten scratch entries into contiguous arrays, one entry per ppi node. */
  ppi_cur = ppi_head; ppi_idx = 0; j = 0;
  while (ppi_cur != NULL) {
    if (ppi_cur->plaintext == NULL) {
      unsigned char hash[MAX_HASH_OUTPUT_LEN] = {0};
      unsigned int hash_len = hex_to_bytes(ppi_cur->hash, sizeof(hash), hash);
      cl_ulong hash_base_index = hash_to_index(hash, hash_len, fa_args->reduction_offset, plaintext_space_total, 0);
      ppi_scratch_entry *e = &scratch->entries[ppi_idx];

      for (i = 0; i < e->num_potential_start_indices; i++, j++) {
        potential_start_indices[j] = e->potential_start_indices[i];
        potential_start_index_positions[j] = e->potential_start_index_positions[i];
        hash_base_indices[j] = hash_base_index;
        ppi_refs[j] = ppi_cur;
      }
    }
    ppi_cur = ppi_cur->next;
    ppi_idx++;
  }

  /* Patch fa_args for single-device use.  Use j (actual filled count) rather
   * than num_potential_start_indices in case another worker cracked a hash
   * concurrently between binary search and this fill loop. */
  fa_args->total_devices = 1;
  fa_args->gpu.device_number = 0;
  fa_args->use_barrier = 0;
  fa_args->potential_start_indices = potential_start_indices;
  fa_args->num_potential_start_indices = j;
  fa_args->potential_start_index_positions = potential_start_index_positions;
  fa_args->hash_base_indices = hash_base_indices;

  double fa_start_abs = get_elapsed(&search_start_time);
  start_timer(&start_time);

  if (pthread_create(&fa_thread, NULL, &host_thread_false_alarm, fa_args)) {
    perror("Failed to create false alarm thread"); exit(-1);
  }
  if (pthread_join(fa_thread, NULL) != 0) {
    perror("Failed to join false alarm thread"); exit(-1);
  }

  /* Process results. */
  for (j = 0; j < fa_args->num_results; j++) {
    if (fa_args->results[j] != 0) {
      char plaintext[MAX_PLAINTEXT_LEN] = {0};
      unsigned int plaintext_len = 0;

      index_to_plaintext(fa_args->results[j], fa_args->charset, charset_len, fa_args->plaintext_len_min, fa_args->plaintext_len_max, plaintext_space_up_to_index, plaintext, &plaintext_len);

      if (fa_args->hash_type == HASH_NTLM) {
        unsigned char hash[16] = {0};
        char hash_hex[(sizeof(hash) * 2) + 1] = {0};
        ntlm_hash(plaintext, plaintext_len, hash);
        if (!bytes_to_hex(hash, sizeof(hash), hash_hex, sizeof(hash_hex)) || (strcmp(hash_hex, ppi_refs[j]->hash) != 0))
          continue;
      } else if (fa_args->hash_type == HASH_NETNTLMV1) {
        unsigned char hash[8] = {0};
        char hash_hex[(sizeof(hash) * 2) + 1] = {0};
        netntlmv1_hash((unsigned char *)plaintext, plaintext_len, hash);
        if (!bytes_to_hex(hash, sizeof(hash), hash_hex, sizeof(hash_hex)) || (strncmp(hash_hex, ppi_refs[j]->hash, 16) != 0)) {
          printf("Found super false positive!: (Net-NTLMv1('%s') == %s) != %s\n", plaintext, hash_hex, ppi_refs[j]->hash);
          continue;
        }
      } else {
        printf("WARNING: CPU code to double-check this cracked hash has not yet been added.  There is a 60%% chance this is a false positive!  A workaround is to use John The Ripper to validate this result(s).\n");
      }

      LOCK_PPI();
      if (ppi_refs[j]->plaintext == NULL) {
        ppi_refs[j]->plaintext = strdup(plaintext);
        ppi_refs[j]->cracked_key_len = (plaintext_len < (unsigned int)sizeof(ppi_refs[j]->cracked_key))
            ? plaintext_len : (unsigned int)sizeof(ppi_refs[j]->cracked_key);
        memcpy(ppi_refs[j]->cracked_key, plaintext, ppi_refs[j]->cracked_key_len);
        ppi_refs[j]->num_precomputed_end_indices = 0;
        FREE(ppi_refs[j]->precomputed_end_indices);
      }
      UNLOCK_PPI();

      if (ppi_refs[j]->plaintext != NULL) {
        pthread_mutex_lock(&potfile_mutex);
        save_cracked_hash(ppi_refs[j], fa_args->hash_type);
        pthread_mutex_unlock(&potfile_mutex);

        if (fa_args->hash_type == HASH_NETNTLMV1) {
          char ptxt_hex[(sizeof(plaintext) * 2) + 1] = {0};
          bytes_to_hex((unsigned char*)plaintext, 7, ptxt_hex, sizeof(ptxt_hex));
          printf("%sHASH CRACKED => %s:1122334455667788:%s%s\n", GREENB, ppi_refs[j]->hash, ptxt_hex, CLR);  fflush(stdout);
        } else {
          printf("%sHASH CRACKED => %s:1122334455667788:%s%s\n", GREENB, (ppi_refs[j]->username != NULL) ? ppi_refs[j]->username : ppi_refs[j]->hash, plaintext, CLR);  fflush(stdout);
        }
      }
    }
  }

  time_delta = get_elapsed(&start_time);
  pthread_mutex_lock(&stats_mutex);
  time_falsealarms += time_delta;
  pthread_mutex_unlock(&stats_mutex);

  seconds_to_human_time(time_str, sizeof(time_str), (unsigned int)time_delta);
  printf("  [GPU #%u] FA started at %.1fs, completed in %s.\n", dev_slot, fa_start_abs, time_str);  fflush(stdout);

  FREE(potential_start_indices);
  FREE(potential_start_index_positions);
  FREE(hash_base_indices);
  FREE(ppi_refs);
  FREE(fa_args->results);
  fa_args->num_results = 0;
}


/* Print a warning to the user if a lot of memory is used by the pre-computed indices. */
void check_memory_usage() {
  uint64_t total_memory = get_total_memory(), num_precompute_bytes = 0;
  double percent_memory_used = 0.0;


  if (total_memory == 0)
    return;

  num_precompute_bytes = total_precomputed_indices_loaded * sizeof(cl_ulong);
  percent_memory_used = ((double)num_precompute_bytes / (double)total_memory) * 100;
  if (percent_memory_used > 65) {
    printf("\n\n\n\t!! WARNING !!\n\n\tThe pre-computed indices take up more than 65%% of total RAM!  This may result in strange failures from clFinish() and other OpenCL functions.  If this happens, either run this lookup with a smaller number of hashes at a time, or do it on a machine with more memory.\n\n\tMemory used by pre-compute indices: %"QUOTE PRIu64"\n\tTotal RAM: %"QUOTE PRIu64"\n\tPercent used: %.1f%%\n\n\n\n", num_precompute_bytes, total_memory, percent_memory_used);
  }
}


/* Free all the potential start indices. */
void clear_potential_start_indices(precomputed_and_potential_indices *ppi) {
  precomputed_and_potential_indices *ppi_cur = ppi;


  while(ppi_cur) {
    FREE(ppi_cur->potential_start_indices);
    FREE(ppi_cur->potential_start_index_positions);
    ppi_cur->num_potential_start_indices = 0;

    ppi_cur = ppi_cur->next;
  }
}


/* Returns the total number of *.rt and *.rtc in all subdirectories of the
 * specified directory. */
unsigned int count_tables(char *dir) {
  DIR *d = NULL;
  struct dirent *de = NULL;
  unsigned int ret = 0, is_file = 0, is_dir = 0;


  d = opendir(dir);
  if (d == NULL) {
    fprintf(stderr, "Failed to open directory %s: %s\n", dir, strerror(errno)); fflush(stderr);
    return 0;
  }

  while ((de = readdir(d)) != NULL) {
#ifdef _WIN32
    struct stat st = {0};
    char path[256] = {0};

    /* The d_type field of the dirent struct is not a POSIX standard, and Windows
     * doesn't support it.  So we fall back to using stat(). */
    snprintf(path, sizeof(path) - 1, "%s\\%s", dir, de->d_name);
    if (stat(path, &st) < 0) {
      fprintf(stderr, "Error: failed to stat() %s: %s.  Continuing anyway...\n", path, strerror(errno));  fflush(stderr);
      is_file = 0;
      is_dir = 0;
    } else {
      is_file = S_ISREG(st.st_mode);
      is_dir = S_ISDIR(st.st_mode);
    }
#else
    {
      struct stat _st = {0};
      char _path[1024] = {0};
      filepath_join(_path, sizeof(_path), dir, de->d_name);
      if (stat(_path, &_st) < 0) {
        is_file = 0;
        is_dir  = 0;
      } else {
        is_file = S_ISREG(_st.st_mode);
        is_dir  = S_ISDIR(_st.st_mode);
      }
    }
#endif

    if (is_file && (str_ends_with(de->d_name, ".rt") || str_ends_with(de->d_name, ".rtc")))
      ret++;
    else if (is_dir && (strcmp(de->d_name, ".") != 0) && (strcmp(de->d_name, "..") != 0)) {
      char subdir_path[1024] = {0};
      filepath_join(subdir_path, sizeof(subdir_path) - 1, dir, de->d_name);
      ret += count_tables(subdir_path);
    }
  }

  closedir(d);
  return ret;
}


/* Free the hashes we loaded from disk or command line. */
void free_loaded_hashes(char **usernames, char **hashes) {
  unsigned int i = 0;

  if (usernames != NULL) {
    for (i = 0; i < num_hashes; i++) {
      FREE(usernames[i]);
    }
    FREE(usernames);
  }

  if (hashes != NULL) {
    for (i = 0; i < num_hashes; i++) {
      FREE(hashes[i]);
    }
    FREE(hashes);
  }
  num_hashes = 0;
}


/* Recursively searches the target directory for the first rainbow table file, and uses its filename to infer
 * the rainbow table parameters. */
void find_rt_params(char *dir_name, rt_parameters *rt_params) {
  char filepath[512] = {0};
  DIR *dir = NULL;
  struct dirent *de = NULL;
  struct stat st;


  dir = opendir(dir_name);
  if (dir == NULL)  /* This directory may not allow the current process permission. */
    return;

  while ((de = readdir(dir)) != NULL) {

    /* Create an absolute path to this entity. */
    filepath_join(filepath, sizeof(filepath), dir_name, de->d_name);

    /* If this is a directory, recurse into it. */
    if ((strcmp(de->d_name, ".") != 0) && (strcmp(de->d_name, "..") != 0) && (stat(filepath, &st) == 0) && S_ISDIR(st.st_mode)) {
      find_rt_params(filepath, rt_params);

      /* If we're searching for rainbowtable parameters, and successfully parsed them
       * in the recursive call, we're done. */
      if ((rt_params != NULL) && rt_params->parsed) {
	closedir(dir); dir = NULL;
	return;
      }

    /* If this is a compressed or uncompressed rainbow table, process it! */
    } else if (str_ends_with(de->d_name, ".rt") || str_ends_with(de->d_name, ".rtc")) {

      /* Try to parse them from this file name.  On success, return immediately
       * (no further processing needed), otherwise continue searching until the
       * first valid set of parameters is found. */
      parse_rt_params(rt_params, de->d_name);
      if (rt_params->parsed) {
	closedir(dir); dir = NULL;
	return;
      }

    }
  }

  closedir(dir); dir = NULL;
}


/* Free the precomputed_hashes linked list. */
void free_precomputed_and_potential_indices(precomputed_and_potential_indices **ppi_head) {
  precomputed_and_potential_indices *ppi = *ppi_head, *ppi_next = NULL;


  while (ppi) {
    ppi_next = ppi->next;

    FREE(ppi->precomputed_end_indices);
    FREE(ppi->potential_start_indices);
    FREE(ppi->potential_start_index_positions);
    FREE(ppi->index_filename);
    ppi->num_potential_start_indices = 0;
    FREE(ppi->plaintext);
    FREE(ppi);

    ppi = ppi_next;
  }
  *ppi_head = NULL;
}


/* Returns the number of CPU cores on this machine. */
unsigned int get_num_cpu_cores() {
#ifdef _WIN32
  SYSTEM_INFO sysinfo = {0};

  GetSystemInfo(&sysinfo);
  return sysinfo.dwNumberOfProcessors;
#else
  return get_nprocs();
#endif
}


/* A host thread which controls each GPU for false alarm checks. */
void *host_thread_false_alarm(void *ptr) {
  thread_args *args = (thread_args *)ptr;
  gpu_dev *gpu = &(args->gpu);
  cl_context context = NULL;
  cl_command_queue queue = NULL;
  cl_kernel kernel = NULL;
  int err = 0;
  char *kernel_path = FALSE_ALARM_KERNEL_PATH, *kernel_name = "false_alarm_check";

  cl_mem hash_type_buffer = NULL, charset_buffer = NULL, plaintext_len_min_buffer = NULL, plaintext_len_max_buffer = NULL, reduction_offset_buffer = NULL, plaintext_space_total_buffer = NULL, plaintext_space_up_to_index_buffer = NULL, device_num_buffer = NULL, total_devices_buffer = NULL, num_start_indices_buffer = NULL, start_indices_buffer = NULL, start_index_positions_buffer = NULL, hash_base_indices_buffer = NULL, output_block_buffer = NULL, exec_block_scaler_buffer = NULL;
  /*cl_mem debug_ulong_buffer = NULL;*/

  cl_ulong *start_indices = NULL, *hash_base_indices = NULL, *plaintext_indices = NULL, *output_block = NULL;
  unsigned int *start_index_positions = NULL;

  unsigned int num_start_indices = 0, num_start_index_positions = 0, num_hash_base_indices = 0, num_plaintext_indices = 0, num_exec_blocks = 0, output_block_len = 0, exec_block = 0, output_block_index = 0, plaintext_indicies_index = 0;
  uint64_t plaintext_space_total = 0;
  cl_ulong plaintext_space_up_to_index[MAX_PLAINTEXT_LEN] = {0};
  size_t gws = 0, kernel_work_group_size = 0, kernel_preferred_work_group_size_multiple = 0;
  /*cl_ulong debug_ulong[128] = {0};*/
  int charset_len = 0;
  if (strcmp(args->charset_name, "byte") == 0) {
    charset_len = 256;
  }
  else {
    charset_len = strlen(args->charset);
  }

  plaintext_space_total = fill_plaintext_space_table(charset_len, args->plaintext_len_min, args->plaintext_len_max, plaintext_space_up_to_index);

  num_start_indices = num_start_index_positions = num_hash_base_indices = num_plaintext_indices = args->num_potential_start_indices;

  start_indices = args->potential_start_indices;
  start_index_positions = args->potential_start_index_positions;
  hash_base_indices = args->hash_base_indices;

  plaintext_indices = calloc(num_plaintext_indices, sizeof(cl_ulong));
  if (plaintext_indices == NULL) {
    fprintf(stderr, "Error while allocating buffers.\n");
    exit(-1);
  }

  /* If we're generating the standard NTLM 8-character tables, use the special
   * optimized kernel instead! */
  if (is_ntlm8(args->hash_type, args->charset, args->plaintext_len_min, args->plaintext_len_max, args->reduction_offset, args->chain_len)) {
    kernel_path = FALSE_ALARM_NTLM8_KERNEL_PATH;
    kernel_name = "false_alarm_check_ntlm8";
    if ((args->gpu.device_number == 0) && (printed_false_alarm_optimized_message == 0)) { /* Only the first thread prints this, and only prints it once. */
      printf("\nNote: optimized NTLM8 kernel will be used for false alarm checks.\n\n"); fflush(stdout);
      printed_false_alarm_optimized_message = 1;
    }
  } else if (is_ntlm9(args->hash_type, args->charset, args->plaintext_len_min, args->plaintext_len_max, args->reduction_offset, args->chain_len)) {
    kernel_path = FALSE_ALARM_NTLM9_KERNEL_PATH;
    kernel_name = "false_alarm_check_ntlm9";
    if ((args->gpu.device_number == 0) && (printed_false_alarm_optimized_message == 0)) { /* Only the first thread prints this, and only prints it once. */
      printf("\nNote: optimized NTLM9 kernel will be used for false alarm checks.\n\n"); fflush(stdout);
      printed_false_alarm_optimized_message = 1;
    }
  }

  /* Load the false alarm kernel, reusing cached objects across tables to avoid
   * repeated JIT compilation.  The cache is populated on first use and kept
   * alive via write-back in table_worker_thread. */
  if (gpu->fa_context == NULL) {
    gpu->fa_context = CLCREATECONTEXT(context_callback, &(gpu->device));
    gpu->fa_queue   = CLCREATEQUEUE(gpu->fa_context, gpu->device);
    load_kernel(gpu->fa_context, 1, &(gpu->device), kernel_path, kernel_name,
                &(gpu->fa_program), &(gpu->fa_kernel), args->hash_type);
  }
  gpu->context = gpu->fa_context;
  gpu->queue   = gpu->fa_queue;
  gpu->kernel  = gpu->fa_kernel;
  gpu->program = gpu->fa_program;

  /* These variables are set so the CLCREATEARG* macros work correctly. */
  context = gpu->context;
  queue   = gpu->queue;
  kernel  = gpu->kernel;

  if ((rc_clGetKernelWorkGroupInfo(kernel, gpu->device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &kernel_work_group_size, NULL) != CL_SUCCESS) || \
      (rc_clGetKernelWorkGroupInfo(kernel, gpu->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &kernel_preferred_work_group_size_multiple, NULL) != CL_SUCCESS)) {
    fprintf(stderr, "Failed to get preferred work group size!\n");
    CLRELEASEKERNEL(gpu->fa_kernel);
    CLRELEASEPROGRAM(gpu->fa_program);
    CLRELEASEQUEUE(gpu->fa_queue);
    CLRELEASECONTEXT(gpu->fa_context);
    gpu->kernel = NULL; gpu->program = NULL;
    gpu->queue  = NULL; gpu->context = NULL;
    pthread_exit(NULL);
    return NULL;
  }

  /* If the user provided a static GWS on the command line, use that.   Otherwise,
   * use the driver's work group size multiplied by the preferred multiple. */
  if (user_provided_gws > 0) {
    gws = user_provided_gws;
    printf("GPU #%u is using user-provided GWS value of %"PRIu64"\n", gpu->device_number, gws);
  } else {
    /*gws = kernel_work_group_size * kernel_preferred_work_group_size_multiple;*/

    /* TODO: fix this so that false alarm checking is done in partitions instead of
     * all at once (this can improve speed).  Currently, when GWS != num_start_indices,
     * lookups don't succeed due to some bug. */
    gws = num_start_indices;

    /* Somehow, on AMD GPUs, the kernel crashes with a message like:
     *
     *   Memory access fault by GPU node-2 (Agent handle: 0x1bb5e00) on address
     *   0x7f4c80b27000. Reason: Page not present or supervisor privilege.
     *
     * A work-around is to set the GWS to the number of start indices and just do it in
     * one pass. */
    if (is_amd_gpu)
      gws = num_start_indices;

    /*printf("GPU #%u is using dynamic GWS: %"PRIu64" (work group) x %"PRIu64" (pref. multiple) = %"PRIu64"\n", gpu->device_number, kernel_work_group_size, kernel_preferred_work_group_size_multiple, gws);*/
  }
  fflush(stdout);


  /* Count the number of times we need to run the kernel. */
  num_exec_blocks = num_start_indices / gws;
  if (num_start_indices % gws != 0)
    num_exec_blocks++;
  //printf("num_exec_blocks: %d, num_start_indices: %d\n", num_exec_blocks, num_start_indices);

  output_block_len = gws;
  output_block = calloc(output_block_len, sizeof(cl_ulong));
  if (output_block == NULL) {
    fprintf(stderr, "Error while allocating output buffer(s).\n");
    exit(-1);
  }

  CLCREATEARG(0, hash_type_buffer, CL_RO, args->hash_type, sizeof(cl_uint));
  CLCREATEARG_ARRAY(1, charset_buffer, CL_RO, args->charset, charset_len + 1);
  CLCREATEARG(2, plaintext_len_min_buffer, CL_RO, args->plaintext_len_min, sizeof(cl_uint));
  CLCREATEARG(3, plaintext_len_max_buffer, CL_RO, args->plaintext_len_max, sizeof(cl_uint));
  CLCREATEARG(4, reduction_offset_buffer, CL_RO, args->reduction_offset, sizeof(cl_uint));
  CLCREATEARG(5, plaintext_space_total_buffer, CL_RO, plaintext_space_total, sizeof(cl_ulong));
  CLCREATEARG_ARRAY(6, plaintext_space_up_to_index_buffer, CL_RO, plaintext_space_up_to_index, MAX_PLAINTEXT_LEN * sizeof(cl_ulong));
  CLCREATEARG(7, device_num_buffer, CL_RO, gpu->device_number, sizeof(cl_uint));
  CLCREATEARG(8, total_devices_buffer, CL_RO, args->total_devices, sizeof(cl_uint));
  CLCREATEARG(9, num_start_indices_buffer, CL_RO, num_start_indices, sizeof(cl_uint));
  CLCREATEARG_ARRAY(10, start_indices_buffer, CL_RO, start_indices, num_start_indices * sizeof(cl_ulong));
  CLCREATEARG_ARRAY(11, start_index_positions_buffer, CL_RO, start_index_positions, num_start_index_positions * sizeof(unsigned int));
  CLCREATEARG_ARRAY(12, hash_base_indices_buffer, CL_RO, hash_base_indices, num_hash_base_indices * sizeof(cl_ulong));
  CLCREATEARG_ARRAY(14, output_block_buffer, CL_WO, output_block, output_block_len * sizeof(cl_ulong));

  for (exec_block = 0; exec_block < num_exec_blocks; exec_block++) {
    unsigned int exec_block_scaler = exec_block * gws;

    CLCREATEARG(13, exec_block_scaler_buffer, CL_RO, exec_block_scaler, sizeof(cl_uint));

    if (args->use_barrier) {
      int barrier_ret = pthread_barrier_wait(&barrier);
      if ((barrier_ret != 0) && (barrier_ret != PTHREAD_BARRIER_SERIAL_THREAD)) {
	fprintf(stderr, "pthread_barrier_wait() failed!\n"); fflush(stderr);
	exit(-1);
      }
    }

    /* Run the kernel and wait for it to finish. */
    CLRUNKERNEL(gpu->queue, gpu->kernel, &gws);
    CLFLUSH(gpu->queue);
    CLWAIT(gpu->queue);

    /* Read the results. */
    CLREADBUFFER(output_block_buffer, output_block_len * sizeof(cl_ulong), output_block);

    output_block_index = 0;
    while ((plaintext_indicies_index < num_plaintext_indices) && (output_block_index < output_block_len))
      plaintext_indices[plaintext_indicies_index++] = output_block[output_block_index++];

    CLFREEBUFFER(exec_block_scaler_buffer);
  }

  /* Set the results so the main thread can access them. */
  args->results = plaintext_indices;
  args->num_results = num_plaintext_indices;  

  FREE(output_block);

  CLFREEBUFFER(hash_type_buffer);
  CLFREEBUFFER(charset_buffer);
  CLFREEBUFFER(plaintext_len_min_buffer);
  CLFREEBUFFER(plaintext_len_max_buffer);
  CLFREEBUFFER(reduction_offset_buffer);
  CLFREEBUFFER(plaintext_space_total_buffer);
  CLFREEBUFFER(plaintext_space_up_to_index_buffer);
  CLFREEBUFFER(device_num_buffer);
  CLFREEBUFFER(total_devices_buffer);
  CLFREEBUFFER(num_start_indices_buffer);
  CLFREEBUFFER(start_indices_buffer);
  CLFREEBUFFER(start_index_positions_buffer);
  CLFREEBUFFER(hash_base_indices_buffer);
  CLFREEBUFFER(output_block_buffer);

  /* fa_context/fa_queue/fa_program/fa_kernel are kept alive for reuse across
   * tables; only the per-call data buffers above are freed here. */

  pthread_exit(NULL);
  return NULL;
}


/* Benchmarks the precompute kernel at several candidate GWS values and
 * returns the fastest one.  Returns 0 on failure or when only one candidate
 * exists (no tuning needed).  Does NOT modify gpu->context/queue/kernel. */
static size_t autotune_precompute_gws(thread_args *args) {
  gpu_dev *gpu = &(args->gpu);
  char *kernel_path = PRECOMPUTE_KERNEL_PATH, *kernel_name = "precompute";
  cl_mem hash_type_buffer = NULL, hash_buffer = NULL, hash_len_buffer = NULL;
  cl_mem charset_buffer = NULL, plen_min_buffer = NULL, plen_max_buffer = NULL;
  cl_mem table_index_buffer = NULL, chain_len_buffer = NULL, dev_num_buffer = NULL;
  cl_mem total_dev_buffer = NULL, scaler_buffer = NULL, out_buffer = NULL;
  cl_ulong *output_block = NULL;
  int err = 0;
  size_t wg = 0, best_gws = 0, base = 0, max_cand = 0, candidates[8] = {0};
  double best_throughput = 0.0;
  unsigned int output_len = 0, num_cands = 0, charset_len = 0, m = 0, j = 0;
  static const unsigned int mults[] = {1, 2, 4, 8, 16};
  unsigned char hash_binary[32] = {0};
  cl_uint hash_binary_len = 0, bench_dev = 0, bench_total = 1, bench_scaler = 0;
  cl_ulong bench_chain_len = 0;
  /* Local aliases required by CLCREATEARG / CLRUNKERNEL macros. */
  cl_context context = NULL;
  cl_command_queue queue = NULL;
  cl_kernel kernel = NULL;

  if (args->hash == NULL)
    return 0;

  hash_binary_len = hex_to_bytes(args->hash, sizeof(hash_binary), hash_binary);

  output_len = args->chain_len / args->total_devices;
  if ((args->chain_len % args->total_devices) != 0)
    output_len++;

  if (is_ntlm8(args->hash_type, args->charset, args->plaintext_len_min,
               args->plaintext_len_max, args->reduction_offset, args->chain_len)) {
    kernel_path = PRECOMPUTE_NTLM8_KERNEL_PATH;
    kernel_name = "precompute_ntlm8";
  } else if (is_ntlm9(args->hash_type, args->charset, args->plaintext_len_min,
                      args->plaintext_len_max, args->reduction_offset, args->chain_len)) {
    kernel_path = PRECOMPUTE_NTLM9_KERNEL_PATH;
    kernel_name = "precompute_ntlm9";
  }

  /* Compile the kernel and store it in gpu so host_thread_precompute can
   * reuse it, avoiding a second expensive JIT compilation. */
  gpu->context = CLCREATECONTEXT(context_callback, &(gpu->device));
  gpu->queue   = CLCREATEQUEUE(gpu->context, gpu->device);
  load_kernel(gpu->context, 1, &(gpu->device), kernel_path, kernel_name,
              &(gpu->program), &(gpu->kernel), args->hash_type);
  context = gpu->context;
  queue   = gpu->queue;
  kernel  = gpu->kernel;

  if (rc_clGetKernelWorkGroupInfo(kernel, gpu->device, CL_KERNEL_WORK_GROUP_SIZE,
                                  sizeof(size_t), &wg, NULL) != CL_SUCCESS) {
    fprintf(stderr, "GWS autotune: failed to query work group size.\n");
    goto done;
  }

  /* Build candidate set: base * {1,2,4,8,16}, cap each to output_len, dedup. */
  base = wg * (size_t)gpu->num_work_units;
  for (m = 0; m < 5; m++) {
    size_t c = base * (size_t)mults[m];
    if (c == 0) continue;
    if (c > (size_t)output_len) c = (size_t)output_len;
    for (j = 0; j < num_cands; j++)
      if (candidates[j] == c) break;
    if (j == num_cands && num_cands < 8)
      candidates[num_cands++] = c;
  }

  if (num_cands <= 1) {
    best_gws = (num_cands == 1) ? candidates[0] : 0;
    fprintf(stderr, "[autotune diag] early-exit: num_cands=%u best_gws=%zu output_len=%u\n",
            num_cands, best_gws, output_len);
    fflush(stderr);
    goto done;
  }

  /* Size output buffer to the largest candidate. */
  max_cand = candidates[0];
  for (j = 1; j < num_cands; j++)
    if (candidates[j] > max_cand) max_cand = candidates[j];

  output_block = calloc(max_cand, sizeof(cl_ulong));
  if (output_block == NULL) {
    fprintf(stderr, "GWS autotune: out of memory.\n");
    goto done;
  }

  charset_len = (strcmp(args->charset_name, "byte") == 0) ?
                256 : (unsigned int)strlen(args->charset);
  /* Use a short chain length for benchmarking — relative GWS throughput is
   * independent of chain depth, so 200 steps is enough while keeping each
   * kernel launch under a millisecond instead of seconds. */
  bench_chain_len = (cl_ulong)(args->chain_len < 200 ? args->chain_len : 200);

  CLCREATEARG(0, hash_type_buffer, CL_RO, args->hash_type, sizeof(cl_uint));
  CLCREATEARG_ARRAY(1, hash_buffer, CL_RO, hash_binary, hash_binary_len);
  CLCREATEARG(2, hash_len_buffer, CL_RO, hash_binary_len, sizeof(cl_uint));
  CLCREATEARG_ARRAY(3, charset_buffer, CL_RO, args->charset, charset_len + 1);
  CLCREATEARG(4, plen_min_buffer, CL_RO, args->plaintext_len_min, sizeof(cl_uint));
  CLCREATEARG(5, plen_max_buffer, CL_RO, args->plaintext_len_max, sizeof(cl_uint));
  CLCREATEARG(6, table_index_buffer, CL_RO, args->table_index, sizeof(cl_uint));
  CLCREATEARG(7, chain_len_buffer, CL_RO, bench_chain_len, sizeof(cl_ulong));
  CLCREATEARG(8, dev_num_buffer, CL_RO, bench_dev, sizeof(cl_uint));
  CLCREATEARG(9, total_dev_buffer, CL_RO, bench_total, sizeof(cl_uint));
  CLCREATEARG(10, scaler_buffer, CL_RO, bench_scaler, sizeof(cl_uint));
  CLCREATEARG_ARRAY(11, out_buffer, CL_WO, output_block, max_cand * sizeof(cl_ulong));

  printf("  Autotuning GWS for GPU #%u...\n", gpu->device_number); fflush(stdout);

  for (m = 0; m < num_cands; m++) {
    size_t gws = candidates[m];
    struct timespec t = {0};
    double elapsed = 0.0, throughput = 0.0;
    int r = 0;

    /* Warm-up run (discarded). */
    CLRUNKERNEL(queue, kernel, &gws);
    CLFLUSH(queue);
    CLWAIT(queue);

    /* Three timed runs. */
    start_timer(&t);
    for (r = 0; r < 3; r++) {
      CLRUNKERNEL(queue, kernel, &gws);
      CLFLUSH(queue);
      CLWAIT(queue);
    }
    elapsed = get_elapsed(&t);
    if (elapsed <= 0.0) elapsed = 1e-9;
    throughput = (double)gws * 3.0 / elapsed;

    if (throughput > best_throughput) {
      best_throughput = throughput;
      best_gws = gws;
    }
  }

  printf("  GWS autotune GPU #%u: selected %zu.\n", gpu->device_number, best_gws);
  fflush(stdout);

done:
  /* context/program/queue/kernel are kept in gpu for host_thread_precompute
   * to reuse; only the temporary benchmark buffers are freed here. */
  CLFREEBUFFER(hash_type_buffer);
  CLFREEBUFFER(hash_buffer);
  CLFREEBUFFER(hash_len_buffer);
  CLFREEBUFFER(charset_buffer);
  CLFREEBUFFER(plen_min_buffer);
  CLFREEBUFFER(plen_max_buffer);
  CLFREEBUFFER(table_index_buffer);
  CLFREEBUFFER(chain_len_buffer);
  CLFREEBUFFER(dev_num_buffer);
  CLFREEBUFFER(total_dev_buffer);
  CLFREEBUFFER(scaler_buffer);
  CLFREEBUFFER(out_buffer);
  free(output_block);
  output_block = NULL;
  return best_gws;
}


/* Pre-warms the FA kernel for one device by JIT-compiling it at startup so the
 * first table's FA call does not pay the ~11.5 s compilation penalty. */
static void prewarm_fa_kernel(thread_args *args) {
  char *kernel_path = FALSE_ALARM_KERNEL_PATH, *kernel_name = "false_alarm_check";
  gpu_dev *gpu = &(args->gpu);
  int err = 0;

  if (is_ntlm8(args->hash_type, args->charset, args->plaintext_len_min,
               args->plaintext_len_max, args->reduction_offset, args->chain_len)) {
    kernel_path = FALSE_ALARM_NTLM8_KERNEL_PATH;
    kernel_name = "false_alarm_check_ntlm8";
  } else if (is_ntlm9(args->hash_type, args->charset, args->plaintext_len_min,
                      args->plaintext_len_max, args->reduction_offset, args->chain_len)) {
    kernel_path = FALSE_ALARM_NTLM9_KERNEL_PATH;
    kernel_name = "false_alarm_check_ntlm9";
  }

  printf("  Pre-warming FA kernel for GPU #%u (%s)...", gpu->device_number, kernel_name);
  fflush(stdout);

  gpu->fa_context = CLCREATECONTEXT(context_callback, &(gpu->device));
  gpu->fa_queue   = CLCREATEQUEUE(gpu->fa_context, gpu->device);
  load_kernel(gpu->fa_context, 1, &(gpu->device), kernel_path, kernel_name,
              &(gpu->fa_program), &(gpu->fa_kernel), args->hash_type);

  printf(" done.\n"); fflush(stdout);
}


/* A host thread which controls each GPU for hash pre-computation. */
void *host_thread_precompute(void *ptr) {
  thread_args *args = (thread_args *)ptr;
  gpu_dev *gpu = &(args->gpu);
  cl_context context = NULL;
  cl_command_queue queue = NULL;
  cl_kernel kernel = NULL;
  int err = 0;
  char *kernel_path = PRECOMPUTE_KERNEL_PATH, *kernel_name = "precompute";

  cl_mem hash_type_buffer = NULL, hash_buffer = NULL, hash_len_buffer = NULL, charset_buffer = NULL, plaintext_len_min_buffer = NULL, plaintext_len_max_buffer = NULL, table_index_buffer = NULL, chain_len_buffer = NULL, device_num_buffer = NULL, total_devices_buffer = NULL, exec_block_scaler_buffer = NULL, output_block_buffer = NULL/*, debug_buffer = NULL*/;

  size_t gws = 0;
  cl_ulong *output = NULL, *output_block = NULL;
  unsigned int output_len = 0, output_block_len = 0, num_exec_blocks = 0, exec_block = 0, output_index = 0, output_block_index = 0;
  /*unsigned int i = 0;*/

  unsigned char hash_binary[32] = {0};
  cl_uint hash_binary_len = 0;


  /* Convert the hash from a hex string to bytes.*/
  hash_binary_len = hex_to_bytes(args->hash, sizeof(hash_binary), hash_binary);

  /* The work size is the chain length divided among the total number of GPUs.  Round
   * up if it doesn't divide evenly; this results in slightly more work being done in
   * order to get complete coverage. */
  output_len = args->chain_len / args->total_devices;
  if ((args->chain_len % args->total_devices) != 0)
    output_len++;

  /* If we're generating the standard NTLM 8-character tables, use the special
   * optimized kernel instead! */
  if (is_ntlm8(args->hash_type, args->charset, args->plaintext_len_min, args->plaintext_len_max, args->reduction_offset, args->chain_len)) {
    kernel_path = PRECOMPUTE_NTLM8_KERNEL_PATH;
    kernel_name = "precompute_ntlm8";
    if ((args->gpu.device_number == 0) && (printed_precompute_optimized_message == 0)) { /* Only the first thread prints this, and only prints it once. */
      printf("\nNote: optimized NTLM8 kernel will be used for precomputation.\n\n"); fflush(stdout);
      printed_precompute_optimized_message = 1;
    }
  } else if (is_ntlm9(args->hash_type, args->charset, args->plaintext_len_min, args->plaintext_len_max, args->reduction_offset, args->chain_len)) {
    kernel_path = PRECOMPUTE_NTLM9_KERNEL_PATH;
    kernel_name = "precompute_ntlm9";
    if ((args->gpu.device_number == 0) && (printed_precompute_optimized_message == 0)) { /* Only the first thread prints this, and only prints it once. */
      printf("\nNote: optimized NTLM9 kernel will be used for precomputation.\n\n"); fflush(stdout);
      printed_precompute_optimized_message = 1;
    }
  }

  /* Load the kernel, reusing the already-compiled objects from autotune if
   * available to avoid a second expensive JIT compilation. */
  if (gpu->context == NULL) {
    gpu->context = CLCREATECONTEXT(context_callback, &(gpu->device));
    gpu->queue = CLCREATEQUEUE(gpu->context, gpu->device);
    load_kernel(gpu->context, 1, &(gpu->device), kernel_path, kernel_name, &(gpu->program), &(gpu->kernel), args->hash_type);
  }

  /* These variables are set so the CLCREATEARG* macros work correctly. */
  context = gpu->context;
  queue = gpu->queue;
  kernel = gpu->kernel;

  if (user_provided_gws > 0) {
    gws = user_provided_gws;
  } else if (gpu->tuned_gws > 0) {
    gws = gpu->tuned_gws;
  } else {
    if (rc_clGetKernelWorkGroupInfo(kernel, gpu->device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &gws, NULL) != CL_SUCCESS) {
      fprintf(stderr, "Failed to get preferred work group size!\n");
      CLRELEASEKERNEL(gpu->kernel);
      CLRELEASEPROGRAM(gpu->program);
      CLRELEASEQUEUE(gpu->queue);
      CLRELEASECONTEXT(gpu->context);
      pthread_exit(NULL);
      return NULL;
    }
    gws = gws * gpu->num_work_units;
  }

  /* In the event that the global work size is larger than the number of outputs we
   * need, cap the GWS. */
  if (gws > output_len) gws = output_len;

  /* Count the number of times we need to run the kernel. */
  num_exec_blocks = output_len / gws;
  if (output_len % gws != 0)
    num_exec_blocks++;

  /*printf("Host thread #%u started; GWS: %zu.\n", gpu->device_number, gws);*/

  /* This will hold the results from this one GPU. */
  output = calloc(output_len, sizeof(cl_ulong));

  /* Holds the results from one kernel exec. */
  output_block_len = gws;
  output_block = calloc(output_block_len, sizeof(cl_ulong));

  if ((output == NULL) || (output_block == NULL)) {
    fprintf(stderr, "Error while allocating output buffer(s).\n");
    exit(-1);
  }

  /* Get the number of compute units in this device. */
  /*get_device_uint(gpu->device, CL_DEVICE_MAX_COMPUTE_UNITS, &(gpu->num_work_units));*/

  int charset_len = 0;
  if (strcmp(args->charset_name, "byte") == 0) {
    charset_len = 256;
  }
  else {
    charset_len = strlen(args->charset);
  }


  CLCREATEARG(0, hash_type_buffer, CL_RO, args->hash_type, sizeof(cl_uint));
  CLCREATEARG_ARRAY(1, hash_buffer, CL_RO, hash_binary, hash_binary_len);
  CLCREATEARG(2, hash_len_buffer, CL_RO, hash_binary_len, sizeof(cl_uint));
  CLCREATEARG_ARRAY(3, charset_buffer, CL_RO, args->charset, charset_len + 1);
  CLCREATEARG(4, plaintext_len_min_buffer, CL_RO, args->plaintext_len_min, sizeof(cl_uint));
  CLCREATEARG(5, plaintext_len_max_buffer, CL_RO, args->plaintext_len_max, sizeof(cl_uint));
  CLCREATEARG(6, table_index_buffer, CL_RO, args->table_index, sizeof(cl_uint));
  cl_ulong chain_len_ulong = (cl_ulong)args->chain_len;
  CLCREATEARG(7, chain_len_buffer, CL_RO, chain_len_ulong, sizeof(cl_ulong));
  CLCREATEARG(8, device_num_buffer, CL_RO, gpu->device_number, sizeof(cl_uint));
  CLCREATEARG(9, total_devices_buffer, CL_RO, args->total_devices, sizeof(cl_uint));
  CLCREATEARG_ARRAY(11, output_block_buffer, CL_WO, output_block, output_block_len * sizeof(cl_ulong));
  /*CLCREATEARG_DEBUG(9, debug_buffer, debug_ptr);*/

  fprintf(stderr, "[precompute diag] charset='%s' charset_len=%d table_index=%u "
          "plaintext_len=%u-%u chain_len=%lu hash=%s gws=%zu num_exec_blocks=%u\n",
          args->charset_name, charset_len, args->table_index,
          args->plaintext_len_min, args->plaintext_len_max,
          (unsigned long)args->chain_len, args->hash, gws, num_exec_blocks);
  fflush(stderr);

  for (exec_block = 0; exec_block < num_exec_blocks; exec_block++) {
    unsigned int exec_block_scaler = exec_block * gws;


    CLCREATEARG(10, exec_block_scaler_buffer, CL_RO, exec_block_scaler, sizeof(cl_uint));

    if (is_amd_gpu) {
      int barrier_ret = pthread_barrier_wait(&barrier);
      if ((barrier_ret != 0) && (barrier_ret != PTHREAD_BARRIER_SERIAL_THREAD)) {
	fprintf(stderr, "pthread_barrier_wait() failed!\n"); fflush(stderr);
	exit(-1);
      }
    }

    /* Run the kernel and wait for it to finish. */
    CLRUNKERNEL(gpu->queue, gpu->kernel, &gws);
    CLFLUSH(gpu->queue);
    CLWAIT(gpu->queue);

    /* Read the results. */
    CLREADBUFFER(output_block_buffer, output_block_len * sizeof(cl_ulong), output_block);

    /* Append this block out output to the total output for this GPU. */
    output_block_index = 0;
    while ((output_index < output_len) && (output_block_index < output_block_len))
      output[output_index++] = output_block[output_block_index++];

    CLFREEBUFFER(exec_block_scaler_buffer);
  }

  /* Set the results so the main thread can access them. */
  args->results = output;
  args->num_results = output_len;

  /*
  printf("GPU %u: ", gpu->device_number);
  for (i = 0; i < output_len; i++) {
    printf("%"PRIu64" ", output[i]);
  }
  printf("\n");
  */

  FREE(output_block);

  CLFREEBUFFER(hash_type_buffer);
  CLFREEBUFFER(hash_buffer);
  CLFREEBUFFER(hash_len_buffer);
  CLFREEBUFFER(charset_buffer);
  CLFREEBUFFER(plaintext_len_min_buffer);
  CLFREEBUFFER(plaintext_len_max_buffer);
  CLFREEBUFFER(table_index_buffer);
  CLFREEBUFFER(chain_len_buffer);
  CLFREEBUFFER(device_num_buffer);
  CLFREEBUFFER(total_devices_buffer);
  CLFREEBUFFER(exec_block_scaler_buffer);
  CLFREEBUFFER(output_block_buffer);
  /*CLFREEBUFFER(debug_buffer);*/

  CLRELEASEKERNEL(gpu->kernel);
  CLRELEASEPROGRAM(gpu->program);
  CLRELEASEQUEUE(gpu->queue);
  CLRELEASECONTEXT(gpu->context);

  pthread_exit(NULL);
  return NULL;
}


void precompute_hash(unsigned int num_devices, thread_args *args, precomputed_and_potential_indices **ppi_head) {
  pthread_t threads[MAX_NUM_DEVICES] = {0};
  char filename[128] = {0}, time_str[128] = {0}, index_data[256] = {0};
  struct timespec start_time = {0};
  unsigned int i = 0, j = 0, output_index = 0;
  int k = 0;
  uint64_t *output = NULL;
  FILE *f = NULL;
  precomputed_and_potential_indices *ppi = NULL;


  /* Set the index data we're looking for (or will create later). */
  snprintf(index_data, sizeof(index_data) - 1, "%s_%s#%d-%d_%d_%d:%s\n", args->hash_name, args->charset_name, args->plaintext_len_min, args->plaintext_len_max, args->table_index, args->chain_len, args->hash); /*ntlm_loweralpha#8-8_0_100:49e5bfaab1be72a6c5236f15736a3e15*/
  fprintf(stderr, "[precompute_hash diag] index_data='%s'\n", index_data);
  fflush(stderr);

  /* Search through the cache and see if we already precomputed the indices for this
   * hash. */
  output = search_precompute_cache(index_data, &output_index, filename, sizeof(filename));

  /* Cache miss... */
  if (output == NULL) {
  
    /* Start the timer for this hash. */
    start_timer(&start_time);

    /* Start one thread to control each GPU. */
    for (i = 0; i < num_devices; i++) {
      if (pthread_create(&(threads[i]), NULL, &host_thread_precompute, &(args[i]))) {
	perror("Failed to create thread");
	exit(-1);
      }
    }

    /* Wait for all threads to finish. */
    for (i = 0; i < num_devices; i++) {
      if (pthread_join(threads[i], NULL) != 0) {
	perror("Failed to join with thread");
	exit(-1);
      }
    }

    num_hashes_precomputed++;

    seconds_to_human_time(time_str, sizeof(time_str), get_elapsed(&start_time));
    printf("  Completed in %s.\n", time_str);  fflush(stdout);
    print_eta_precompute();

    /* Create one output array to hold all the results. */
    output = calloc(args[0].num_results * num_devices, sizeof(uint64_t));
    if (output == NULL) {
      fprintf(stderr, "Error allocating buffer for GPU results.\n");
      exit(-1);
    }

    /*
      The results end up spread out like this across many GPUs:

      GPU 0: 100 94 88 82 76 70 64 58 52 46 40 34 28 22 16 10 4 
      GPU 1: 99 93 87 81 75 69 63 57 51 45 39 33 27 21 15 9 3 
      GPU 2: 98 92 86 80 74 68 62 56 50 44 38 32 26 20 14 8 2 
      GPU 3: 97 91 85 79 73 67 61 55 49 43 37 31 25 19 13 7 1 
      GPU 4: 96 90 84 78 72 66 60 54 48 42 36 30 24 18 12 6 0 
      GPU 5: 95 89 83 77 71 65 59 53 47 41 35 29 23 17 11 5 0 

      Below, we collate the results into a single array containing "100 99 98 [...]".
    */
    for (i = 0; i < args[0].num_results; i++) {
      for (j = 0; j < num_devices; j++) {
	output[output_index] = args[j].results[i];
	output_index++;
      }
    }

    /* Now that pulled all the GPU results into one array, free them. */
    for (i = 0; i < num_devices; i++) {
      FREE(args[i].results);
      args[i].num_results = 0;
    }

    /* We may have a few extra indices in the array at the end, if the chain length
     * is not divisible by the number of GPUs.  In that case, we simply truncate the
     * end of the array. */
    if (output_index >= args[0].chain_len - 1)
      output_index = args[0].chain_len -1;
    else { /* Sanity check: this should never happen... */
      fprintf(stderr, "Error: output_index < chain_len - 1!: %u < %u\n", output_index, args[0].chain_len - 1);
      exit(-1);
    }

    /* Reverse the output buffer.
     * TODO: this logic can be merged in, above, to simplify. */
    {
      uint64_t *tmp = calloc(output_index, sizeof(uint64_t));
      if (tmp == NULL) {
	fprintf(stderr, "Failed to create temp buffer.\n");
	exit(-1);
      }

      for (i = 0; i < output_index; i++)
	tmp[i] = output[output_index - i - 1];

      FREE(output);
      output = tmp;
    }

    /* Ensure we didn't get all zeros. */
    for (k = 0; k < output_index; k++)
      if (output[k] != 0)
	break;

    if (k == output_index) {
      fprintf(stderr, "Error: all zeros in precomputation!\n");
      exit(-1);
    }

    /* Search for the first unused filename in the space of rcracki.precalc.[0-1048576]. */
    for (i = 0; i < 1048576; i++) {
      int fd = -1;

      snprintf(filename, sizeof(filename) - 1, "rcracki.precalc.%d", i);

      /* Create a file for writing with permissions of 0600. */
      fd = open(filename, O_CREAT | O_EXCL | O_WRONLY | O_BINARY, S_IRUSR | S_IWUSR);

      if (fd != -1) { /* On success, convert to a file pointer. */
	f = fdopen(fd, "wb");
	break;
      }
    }

    if (f == NULL) {
      fprintf(stderr, "Error: could not create any precalc file (rcracki.precalc.[0-1048576])\n");
      exit(-1);
    }

    /* Ok, so it turns out that we generated the array backwards.  Oh well.  We will
     * just iterate backwards here to compensate. */
    /*for (k = output_index - 1; k >= 0; k--)
      fwrite(&(output[k]), sizeof(cl_ulong), 1, f);*/

    for (k = 0; k < output_index; k++)
      fwrite(&(output[k]), sizeof(cl_ulong), 1, f);

    FCLOSE(f);

    /* Now create the rcracki.precalc.?.index file. */
    strncat(filename, ".index", sizeof(filename) - 1);
    f = fopen(filename, "wb");
    if (f == NULL) {
      fprintf(stderr, "Error while creating file: %s\n", filename);
      exit(-1);
    } else {
      fwrite(index_data, sizeof(char), strlen(index_data), f);
      FCLOSE(f);
    }

  } else {
    num_hashes_precomputed_total--;
    printf("Using cached pre-computed indices for hash %s.\n", args->hash);  fflush(stdout);
  }

  total_precomputed_indices_loaded += output_index;

  /*
  printf("output_index: %u\nFinal array: ", output_index);

  for (i = 0; i < output_index; i++)
    printf("%"PRIu64" ", output[i]);
  printf("\n");

  printf("\nFinal array hex: ");

  for (i = 0; i < output_index; i++)
    printf("%08"PRIx64" ", output[i]);
  printf("\n");
  */

  /* Time to store the precomputed indices.  If no head exists in the linked list... */
  if (*ppi_head == NULL) {
    *ppi_head = calloc(1, sizeof(precomputed_and_potential_indices));
    if (*ppi_head == NULL) {
      fprintf(stderr, "Error allocating buffer for precomputed indices.\n");
      exit(-1);
    }
    ppi = *ppi_head;
  } else {
    ppi = *ppi_head;
    while (ppi->next != NULL)
      ppi = ppi->next;
    ppi->next = calloc(1, sizeof(precomputed_and_potential_indices));
    if (ppi->next == NULL) {
      fprintf(stderr, "Error allocating buffer for precomputed indices.\n");
      exit(-1);
    }
    ppi = ppi->next;
  }

  ppi->username = args->username;
  ppi->hash = args->hash;
  ppi->num_precomputed_end_indices = output_index;

  ppi->precomputed_end_indices = calloc(ppi->num_precomputed_end_indices, sizeof(cl_ulong));
  if (ppi->precomputed_end_indices == NULL) {
    fprintf(stderr, "Error allocating index buffer for precomputed indices.\n");
    exit(-1);
  }

  /* Store the precomputed indices into the array. */
  for (i = 0; i < ppi->num_precomputed_end_indices; i++)
    ppi->precomputed_end_indices[i] = output[i];

  /* Set the filename, so it can be deleted if the hash is cracked later. */
  ppi->index_filename = strdup(filename);

  FREE(output);
}


void _preloading_thread(char *rt_dir) {
  DIR *dir = NULL;
  struct dirent *de = NULL;
  struct stat st;
  char filepath[512];
  unsigned int local_tables = 0, local_dirs = 0, local_skipped = 0, local_failed = 0;


  memset(&st, 0, sizeof(st));
  memset(filepath, 0, sizeof(filepath));

  fprintf(stderr, "[preloader] Scanning: %s (max_preload_num=%u)\n", rt_dir, max_preload_num);  fflush(stderr);

  dir = opendir(rt_dir);
  if (dir == NULL) {  /* This directory may not allow the current process permission. */
    fprintf(stderr, "[preloader] opendir failed for %s: %s\n", rt_dir, strerror(errno));  fflush(stderr);
    return;
  }

  while ((de = readdir(dir)) != NULL) {
    if (stop_preloading) break;

    /* Create an absolute path to this entity. */
    filepath_join(filepath, sizeof(filepath), rt_dir, de->d_name);

    /* If this is a directory, recurse into it. */
    if ((strcmp(de->d_name, ".") != 0) && (strcmp(de->d_name, "..") != 0) &&
        (stat(filepath, &st) != 0 ? (fprintf(stderr, "[preloader] stat failed: %s (%s)\n", filepath, strerror(errno)), fflush(stderr), 0) : 1) &&
        S_ISDIR(st.st_mode)) {
      local_dirs++;
      _preloading_thread(filepath);

    /* If this is a compressed or uncompressed rainbow table, load it! */
    } else if (str_ends_with(de->d_name, ".rt") || str_ends_with(de->d_name, ".rtc")) {
      cl_ulong *rainbow_table = NULL;
      unsigned int num_chains = 0, is_uncompressed_table = 0;
      struct timespec start_time_io = {0};


      if (str_ends_with(de->d_name, ".rtc")) {
	int ret = 0;

	start_timer(&start_time_io);    /* For loading the table only. */
	if ((ret = rtc_decompress(filepath, &rainbow_table, &num_chains)) != 0) {
	  fprintf(stderr, "Error while decompressing RTC table %s: %d\n", filepath, ret);
	  exit(-1);
	}
	time_io += get_elapsed(&start_time_io);
      } else {
	FILE *f = NULL;

	is_uncompressed_table = 1;
	start_timer(&start_time_io);    /* For loading the table only. */
	f = fopen(filepath, "rb");
	if (f != NULL) {
	  long file_size = get_file_size(f);

	  if ((file_size % (sizeof(cl_ulong) * 2) == 0) && (file_size > 0)) {
	    unsigned int num_longs = file_size / sizeof(cl_ulong);

	    rainbow_table = calloc(num_longs, sizeof(cl_ulong));
	    if (rainbow_table == NULL) {
	      fprintf(stderr, "Failed to allocate %"PRIu64" bytes for rainbow table!: %s\n", num_longs * sizeof(cl_ulong), filepath);
	      exit(-1);
	    }

	    if (fread(rainbow_table, sizeof(cl_ulong), num_longs, f) != num_longs) {
	      fprintf(stderr, "Error while reading rainbow table: %s\n", strerror(errno));
	      exit(-1);
	    }

	    time_io += get_elapsed(&start_time_io);
	    num_chains = num_longs / 2;
	  } else {
	    fprintf(stderr, "[preloader] Bad size (%ld) for: %s\n", file_size, filepath);  fflush(stderr);
	    local_failed++;
	  }

	  FCLOSE(f);
	} else {
	  fprintf(stderr, "[preloader] fopen failed: %s (%s)\n", filepath, strerror(errno));  fflush(stderr);
	  local_failed++;
	}
      }

      if (rainbow_table != NULL) {
	unsigned int skip_table = 0;


	/* If the table is uncompressed (*.rt), then there's a possibility its unsorted on accident.  We will
	 * verify them first to make sure. */
	if (is_uncompressed_table == 1) {
	  if (!verify_rainbowtable(rainbow_table, num_chains, VERIFY_TABLE_TYPE_LOOKUP, 0, 0, NULL)) {
	    fprintf(stderr, "\nError: %s is not a valid table suitable for lookups!  (Hint: it may not be sorted.)  Skipping...\n\n", filepath);  fflush(stderr);
	    FREE(rainbow_table);
	    skip_table = 1; /* Skip further processing on this table only. */
	    local_skipped++;
	  }
	}

	if (!skip_table) {
	  preloaded_table *pt = calloc(1, sizeof(preloaded_table));
	  if (pt == NULL) {
	    printf("Failed to allocate memory for preload_table.\n");
	    exit(-1);
	  }

	  /* Set the file path, rainbow table, and number of chains in the newest entry of the preload list. */
	  pt->filepath = strdup(filepath);
	  pt->rainbow_table = rainbow_table;
	  pt->num_chains = num_chains;

	  /* Lock the preloading system, since we're modifying shared structures. */
	  pthread_mutex_lock(&preloaded_tables_lock);

	  /* Increase the counter of preloaded tables. */
	  num_preloaded_tables_available++;

	  /* If the list is empty, add the newest entry as the head. */
	  if (preloaded_table_list == NULL)
	    preloaded_table_list = pt;
	  else { /* The list isn't empty, so traverse it to the end, and append this entry. */
	    preloaded_table *ptr = preloaded_table_list;
	    while (ptr->next != NULL)
	      ptr = ptr->next;

	    ptr->next = pt;
	  }

	  /* Tell the main thread that we have a table available. */
	  pthread_cond_signal(&condition_wait_for_tables);

	  /* If we preloaded the maximum number of tables, wait for the main thread to consume at least one
	   * before preloading more. */
	  if ((num_preloaded_tables_available >= max_preload_num) && !stop_preloading) {
	    fprintf(stderr, "[preloader] Buffer full: available=%u >= max_preload_num=%u, blocking (local_tables=%u)\n",
	            num_preloaded_tables_available, max_preload_num, local_tables);  fflush(stderr);
	  }
	  while ((num_preloaded_tables_available >= max_preload_num) && !stop_preloading)
	    pthread_cond_wait(&condition_continue_loading_tables, &preloaded_tables_lock);
	  if (stop_preloading) {
	    fprintf(stderr, "[preloader] stop_preloading set mid-wait in %s (%u loaded, %u subdirs, %u skipped, %u failed)\n", rt_dir, local_tables, local_dirs, local_skipped, local_failed);  fflush(stderr);
	    pthread_mutex_unlock(&preloaded_tables_lock);
	    closedir(dir); dir = NULL;
	    return;
	  }

	  /* Release the preloading system lock. */
	  pthread_mutex_unlock(&preloaded_tables_lock);
	  local_tables++;
	  tables_preloaded_count++;
	}
      }
    } else if ((strcmp(de->d_name, ".") != 0) && (strcmp(de->d_name, "..") != 0) &&
               !str_ends_with(de->d_name, ".rt") && !str_ends_with(de->d_name, ".rtc")) {
      /* Entry that is neither a directory (by stat) nor a .rt/.rtc file — could be a
       * symlink-to-directory whose name doesn't end in .rt, or an unrelated file. */
    }
  }

  if (stop_preloading)
    fprintf(stderr, "[preloader] stop_preloading set, exiting early from %s (%u loaded, %u subdirs, %u skipped, %u failed)\n", rt_dir, local_tables, local_dirs, local_skipped, local_failed);
  else
    fprintf(stderr, "[preloader] Finished %s: %u tables loaded, %u subdirs, %u skipped, %u failed\n", rt_dir, local_tables, local_dirs, local_skipped, local_failed);
  fflush(stderr);

  closedir(dir); dir = NULL;
}


/* The thread which preloads tables in the background while the main thread performs binary searching & false
 * alarm checks. */
void *preloading_thread(void *ptr) {
  char *xrt_dir = ((preloading_thread_args *)ptr)->rt_dir;
  char rt_dir[512];


  memset(rt_dir, 0, sizeof(rt_dir));

  /* Copy the rainbow table path from the heap to the local stack, then free the source. */
  strncpy(rt_dir, xrt_dir, sizeof(rt_dir) - 1);
  free(xrt_dir); xrt_dir = ((preloading_thread_args *)ptr)->rt_dir = NULL;

  _preloading_thread(rt_dir);
  fprintf(stderr, "[preloader] Preloading complete: %u tables total. Setting table_loading_complete=1.\n", tables_preloaded_count);  fflush(stderr);

  /* We've reached the end of all the tables, so tell the main thread. */
  pthread_mutex_lock(&preloaded_tables_lock);
  table_loading_complete = 1;
  pthread_cond_broadcast(&condition_wait_for_tables);
  pthread_mutex_unlock(&preloaded_tables_lock);
  return NULL;
}


/* Given the number of hashes processed out of the total, prints the estimated time left to
 * completion. */
void print_eta_precompute() {
  char eta_str[64] = {0};

  strncpy(eta_str, "Unknown", sizeof(eta_str) - 1);
  if ((num_hashes_precomputed > 0) && (num_hashes_precomputed_total >= num_hashes_precomputed)) {
    double seconds_per_hash = (double)(get_elapsed(&precompute_start_time) / (double)num_hashes_precomputed);
    unsigned int num_hashes_left = num_hashes_precomputed_total - num_hashes_precomputed;
    unsigned int num_seconds_left = num_hashes_left * seconds_per_hash;

    seconds_to_human_time(eta_str, sizeof(eta_str), num_seconds_left);
  }
  printf("  Estimated time to complete pre-computation (at most): %s\n\n", eta_str); fflush(stdout);
}


/* Given the number of tables processed out of the total, prints the estimated time left to
 * completion. */
void print_eta_search(unsigned int num_tables_processed, unsigned int num_tables_total) {
  char eta_str[64] = {0};
  char elapsed_str[64] = {0};
  unsigned int pct = (num_tables_total > 0) ? (num_tables_processed * 100 / num_tables_total) : 0;
  double elapsed = get_elapsed(&search_start_time);

  seconds_to_human_time(elapsed_str, sizeof(elapsed_str), (unsigned int)elapsed);
  strncpy(eta_str, "unknown", sizeof(eta_str) - 1);
  if ((num_tables_processed > 0) && (num_tables_total >= num_tables_processed)) {
    double seconds_per_table = elapsed / (double)num_tables_processed;
    unsigned int num_tables_left = num_tables_total - num_tables_processed;
    unsigned int num_seconds_left = (unsigned int)(num_tables_left * seconds_per_table);

    seconds_to_human_time(eta_str, sizeof(eta_str), num_seconds_left);
  }
  printf("\n  Progress: %u/%u tables (%u%%) | Elapsed: %s | ETA: %s\n\n", num_tables_processed, num_tables_total, pct, elapsed_str, eta_str);
  fflush(stdout);
}


void print_usage_and_exit(char *prog_name, int exit_code) {
#ifdef _WIN32
  char *dir1 = "D:\\rt_ntlm\\";
  char *dir2 = "C:\\Users\\jsmith\\Desktop\\";
#else
  char *dir1 = "/export/rt_ntlm/";
  char *dir2 = "/home/user/";
#endif

  fprintf(stderr, "%sUsage:%s %s rainbow_table_directory ( single_hash | filename_with_many_hashes.txt | -ntlmv1 full_capture | -ntlmv1-file captures.txt ) [-gws GWS] [-disable-platform N]\n\n", WHITEB, CLR, prog_name);
  fprintf(stderr, "    %s-gws GWS%s    (Optional) Sets the global work size for each GPU.  This can significantly affect the speed.  To tune this setting, start with multiplying the max compute units by the max work group size (both are reported on program start-up).  Then increase/decrease the value and time the results.  For example, if the max compute units is 20, and the max work group size is 1024, try using 20 x 1024 = 20480, then 20480 - 1024 = 19456, 20480 - 2048 = 18432, 2048 + 1024 = 21504, etc.  If you find a value that works better than the automatic setting, please report your findings at: https://github.com/jtesta/rainbowcrackalack/issues\n\n", WHITEB, CLR);
  fprintf(stderr, "    %s-disable-platform N%s    (Optional) Disables a platform from being used (platform numbers are reported on program start-up).  Useful when experiencing strange problems on mixed-GPU systems.  Try disabling each platform one at a time and see if the program behaves normally.\n\n\n", WHITEB, CLR);
  fprintf(stderr, "%sExamples:%s\n    %s %s 64f12cddaa88057e06a81b54e73b949b\n    %s %s %shashes_one_per_line.txt\n    %s %s %spwdump.txt\n\n", WHITEB, CLR, prog_name, dir1, prog_name, dir1, dir2, prog_name, dir1, dir2);
  exit(exit_code);
}


/* Helper function for rt_binary_search(). */
unsigned int _rt_binary_search(cl_ulong *rainbow_table, unsigned int low, unsigned int high, cl_ulong search_index, cl_ulong *start) {
  unsigned int chain = 0;


  /*printf("_rt_binary_search(%u, %u, %lu)\n", low, high, search_index);*/
  if (high - low <= 8) {
    for (chain = low; chain < high; chain++) {
      if (search_index == rainbow_table[(chain * 2) + 1]) {
	*start = rainbow_table[chain * 2];
	/*printf("\nbinary search: found %lu at %u (between %u and %u)\n", *start, chain, low, high);*/
	return 1;
      }
    }
  } else {
    chain = ((high - low) / 2) + low;
    if (search_index >= rainbow_table[(chain * 2) + 1])
      return _rt_binary_search(rainbow_table, chain, high, search_index, start);
    else
      return _rt_binary_search(rainbow_table, low, chain, search_index, start);
  }

  return 0;
}


void *rt_binary_search_thread(void *ptr) {
  search_thread_args *args = (search_thread_args *)ptr;
  precomputed_and_potential_indices *ppi_cur = args->ppi_head;
  unsigned int i = 0, ppi_idx = 0;
  cl_ulong start = 0;


  while (ppi_cur != NULL) {
    if (ppi_cur->plaintext == NULL) {
      for (i = 0 + args->thread_number; i < ppi_cur->num_precomputed_end_indices; i += args->total_threads) {
	if (_rt_binary_search(args->rainbow_table, 0, args->num_chains, ppi_cur->precomputed_end_indices[i], &start)) {
	  scratch_add_match(args->scratch, ppi_idx, start, i);
	}
      }
    }
    ppi_cur = ppi_cur->next;
    ppi_idx++;
  }

  pthread_exit(NULL);
  return NULL;
}


/* Rainbow table binary search.  Searches a table's end indices for any matches with
 * precomputed end indices.  If/when matches are found, the corresponding start indices
 * are added to the precomputed_and_potential_indices's potential_start_indices
 * array. */
void rt_binary_search(cl_ulong *rainbow_table, unsigned int num_chains, precomputed_and_potential_indices *ppi_head, unsigned int num_threads, worker_scratch *scratch) {
  struct timespec start_time_searching = {0};
  char time_searching_str[64] = {0};
  pthread_t *threads = NULL;
  search_thread_args *args = NULL;
  unsigned int i = 0;
  double s_time = 0;


  start_timer(&start_time_searching);
  args = calloc(num_threads, sizeof(search_thread_args));
  threads = calloc(num_threads, sizeof(pthread_t));
  if ((args == NULL) || (threads == NULL)) {
    fprintf(stderr, "Failed to create thread/args for searching.\n");
    exit(-1);
  }

  for (i = 0; i < num_threads; i++) {
    args[i].thread_number = i;
    args[i].total_threads = num_threads;
    args[i].rainbow_table = rainbow_table;
    args[i].num_chains = num_chains;
    args[i].ppi_head = ppi_head;
    args[i].scratch = scratch;

    if (pthread_create(&(threads[i]), NULL, &rt_binary_search_thread, &(args[i]))) {
      perror("Failed to create thread");
      exit(-1);
    }
  }

  for (i = 0; i < num_threads; i++) {
    if (pthread_join(threads[i], NULL) != 0) {
      perror("Failed to join with thread");
      exit(-1);
    }
  }

  s_time = get_elapsed(&start_time_searching);
  seconds_to_human_time(time_searching_str, sizeof(time_searching_str), s_time);
  printf("  Table searched in %s.\n", time_searching_str);  fflush(stdout);

  pthread_mutex_lock(&stats_mutex);
  time_searching += s_time;
  pthread_mutex_unlock(&stats_mutex);

  FREE(args);
  FREE(threads);
}


void save_cracked_hash(precomputed_and_potential_indices *ppi, unsigned int hash_type) {
  FILE *jtr_file = fopen(jtr_pot_filename, "ab"), *hashcat_file = fopen(hashcat_pot_filename, "ab");
  unsigned int hash_len = 0, plaintext_len = 0;
  if (hash_type == HASH_NETNTLMV1) {
    hash_len = 8;
    plaintext_len = 8;
  }
  else {
    hash_len = strlen(ppi->hash);
    plaintext_len = strlen(ppi->plaintext);
  }
  char *dot_pos = strrchr(ppi->index_filename, '.');


  if (jtr_file == NULL) {
    fprintf(stderr, "Error: could not open pot file for writing: %s: %s\n", jtr_pot_filename, strerror(errno));
    exit(-1);
  } else if (hashcat_file == NULL) {
    fprintf(stderr, "Error: could not open pot file for writing: %s: %s\n", hashcat_pot_filename, strerror(errno));
    exit(-1);
  }

  /* The JTR pot file format requires NTLM hashes to be prepended with "$NT$". */
  if ((hash_type == HASH_NTLM) && (fwrite("$NT$", sizeof(char), 4, jtr_file) != 4)) {
    fprintf(stderr, "Error while writing to pot file: %s\n", strerror(errno));
    exit(-1);
  }

  if (fwrite(ppi->hash, sizeof(char), hash_len, jtr_file) != hash_len) {
    fprintf(stderr, "Error while writing to pot file: %s\n", strerror(errno));
    exit(-1);
  } else if (fwrite(ppi->hash, sizeof(char), hash_len, hashcat_file) != hash_len) {
    fprintf(stderr, "Error while writing to pot file: %s\n", strerror(errno));
    exit(-1);
  }

  if (fwrite(":", sizeof(char), 1, jtr_file) != 1) {
    fprintf(stderr, "Error while writing to pot file: %s\n", strerror(errno));
    exit(-1);
  } else if (fwrite(":", sizeof(char), 1, hashcat_file) != 1) {
    fprintf(stderr, "Error while writing to pot file: %s\n", strerror(errno));
    exit(-1);
  }

  if (fwrite(ppi->plaintext, sizeof(char), plaintext_len, jtr_file) != plaintext_len) {
    fprintf(stderr, "Error while writing to pot file: %s\n", strerror(errno));
    exit(-1);
  } else if (fwrite(ppi->plaintext, sizeof(char), plaintext_len, hashcat_file) != plaintext_len) {
    fprintf(stderr, "Error while writing to pot file: %s\n", strerror(errno));
    exit(-1);
  }

  if (fwrite("\n", sizeof(char), 1, jtr_file) != 1) {
    fprintf(stderr, "Error while writing to pot file: %s\n", strerror(errno));
    exit(-1);
  } else if (fwrite("\n", sizeof(char), 1, hashcat_file) != 1) {
    fprintf(stderr, "Error while writing to pot file: %s\n", strerror(errno));
    exit(-1);
  }

  FCLOSE(jtr_file);
  FCLOSE(hashcat_file);

  /* Delete the index file containing information about the precomputed indices.  Since
   * this hash was cracked, this is no longer needed. */
  if (unlink(ppi->index_filename) != 0) {
    fprintf(stderr, "Error while deleting precompute index file: %s: %s\n", ppi->index_filename, strerror(errno));
    /*exit(-1);*/
  }

  /* Truncate the ".index" off the end of the filename; this forms the precomputation
   * filename. */
  *dot_pos = '\0';
  if (unlink(ppi->index_filename) != 0) {
    fprintf(stderr, "Error while deleting precompute file: %s: %s\n", ppi->index_filename, strerror(errno));
    /*exit(-1);*/
  }

  pthread_mutex_lock(&stats_mutex);
  num_cracked++;
  num_falsealarms--;
  pthread_mutex_unlock(&stats_mutex);
}


/* Searches the precompute cache for matching index data.  If found, an array of
 * indices is returned, num_indices set to the array size, and the filename buffer
 * is set to the *.index cache file. */
cl_ulong *search_precompute_cache(char *index_data, unsigned int *num_indices, char *filename, unsigned int filename_size) {
  char buf[256] = {0};
  int file_size = 0;
  DIR *d = NULL;
  struct dirent *de = NULL;
  FILE *f = NULL;
  cl_ulong *ret = NULL;


  *num_indices = 0;
  memset(filename, 0, filename_size);


  /* Go through all *.index files in the current directory and find any that match
   * the hash passed to us.  If found, we already pre-computed the values. */
  d = opendir(".");
  if (d == NULL) {
    fprintf(stderr, "Can't open current directory.\n");
    exit(-1);
  }
  while ((de = readdir(d)) != NULL) {
    if (str_ends_with(de->d_name, ".index")) {
      /*printf("Looking at %s\n", de->d_name);*/

      /* Open this *.index file. */
      f = fopen(de->d_name, "rb");
      if (f == NULL) {
	fprintf(stderr, "Failed to open %s for reading.\n", de->d_name);
	exit(-1);
      }

      file_size = get_file_size(f);

      /* Read the index data.*/
      if ((file_size >= sizeof(buf)) || (fread(buf, sizeof(char), file_size, f) != file_size)) {
	fprintf(stderr, "Failed to read index data: %s\n", strerror(errno));
	exit(-1);
      }

      FCLOSE(f);

      /* We found an index file that matches all our parameters.  Open its related
       * file containing precomputed indices. */
      if (strcmp(index_data, buf) == 0) {

	/* Set the filename to the *.index file for the caller. */
	strncpy(filename, de->d_name, filename_size - 1);
	de->d_name[strlen(de->d_name) - 6] = '\0';

	f = fopen(de->d_name, "rb");
	if (f == NULL) {
	  fprintf(stderr, "Failed to open precomputed index file: %s\n", de->d_name);
	  exit(-1);
	}

	file_size = get_file_size(f);

	if (file_size % sizeof(cl_ulong) != 0) {
	  fprintf(stderr, "Precomputed indices file is not a multiple of %"PRIu64": %u\n", sizeof(cl_ulong), file_size);
	  exit(-1);
	}

	*num_indices = file_size / sizeof(cl_ulong);

	ret = calloc(*num_indices, sizeof(cl_ulong));
	if (ret == NULL) {
	  fprintf(stderr, "Failed to create indices buffer.\n");
	  exit(-1);
	}

	if (fread(ret, sizeof(cl_ulong), *num_indices, f) != *num_indices) {
	  fprintf(stderr, "Failed to read indices file.\n");
	  exit(-1);
	}
	FCLOSE(f);

	break;
      }
    }
  }
  closedir(d); d = NULL;  
  return ret;
}


/* Returns a preloaded_table entry, or NULL if no more tables are left to process.  The caller must
 * free it and all member variables. */
preloaded_table *get_preloaded_table() {
  preloaded_table *ret = NULL;

  pthread_mutex_lock(&preloaded_tables_lock);

  /* If no tables have been preloaded yet, wait until at least one becomes available. */
  while ((num_preloaded_tables_available == 0) && (table_loading_complete == 0)) {
    fprintf(stderr, "[get_table] Waiting: available=%u tlc=%u\n",
            num_preloaded_tables_available, table_loading_complete);  fflush(stderr);
    pthread_cond_wait(&condition_wait_for_tables, &preloaded_tables_lock);
    fprintf(stderr, "[get_table] Woke: available=%u tlc=%u\n",
            num_preloaded_tables_available, table_loading_complete);  fflush(stderr);
  }

  /* Return the head of the list. */
  ret = preloaded_table_list;
  if (ret == NULL)
    fprintf(stderr, "[get_table] Returning NULL: available=%u tlc=%u list=%s\n",
            num_preloaded_tables_available, table_loading_complete,
            preloaded_table_list == NULL ? "empty" : "non-empty");  fflush(stderr);

  /* If the head of the list isn't NULL, advance it by one. */
  if (preloaded_table_list != NULL) {
    preloaded_table_list = preloaded_table_list->next;

    if (num_preloaded_tables_available > 0)
      num_preloaded_tables_available--;

    /* Wake up the preloading thread if its waiting because it loaded the max.  Now that we're
     * consuming one table, it can load the next concurrently. */
    pthread_cond_signal(&condition_continue_loading_tables);
  }

  pthread_mutex_unlock(&preloaded_tables_lock);
  return ret;
}


/* One worker in the parallel table-search pool.  Accumulates binary-search
 * results from FA_BATCH_SIZE tables before triggering one GPU FA launch, which
 * increases kernel occupancy ~4× vs. launching per-table. */
void *table_worker_thread(void *ptr) {
  worker_thread_args *wargs = (worker_thread_args *)ptr;
  worker_scratch *scratch = alloc_worker_scratch(wargs->ppi_head);
  unsigned int tables_in_scratch = 0;

  while (1) {
    preloaded_table *pt = NULL;
    cl_ulong *rainbow_table = NULL;
    unsigned int num_chains = 0;
    unsigned int dev_slot = 0;
    thread_args fa_args;
    int flush_fa = 0;

    pthread_mutex_lock(&stats_mutex);
    if (all_cracked) {
      fprintf(stderr, "[worker %u] Exiting: all_cracked was set\n", wargs->worker_id);  fflush(stderr);
      pthread_mutex_unlock(&stats_mutex); break;
    }
    pthread_mutex_unlock(&stats_mutex);

    pt = get_preloaded_table();
    /* Capture the end-of-tables sentinel BEFORE FREE(pt) nulls the pointer. */
    int no_more_tables = (pt == NULL);
    if (no_more_tables) {
      /* No more tables; flush any accumulated BS results. */
      flush_fa = (tables_in_scratch > 0);
    } else {
      rainbow_table = pt->rainbow_table;
      num_chains = pt->num_chains;
      printf("  [worker %u] Processing: %s\n", wargs->worker_id, pt->filepath);  fflush(stdout);
      FREE(pt->filepath);
      FREE(pt);  /* pt is now NULL — that is fine; no_more_tables is already captured */

      /* Accumulate BS results into scratch without clearing between tables. */
      rt_binary_search(rainbow_table, num_chains, wargs->ppi_head, wargs->bs_threads, scratch);
      FREE(rainbow_table);
      tables_in_scratch++;

      /* Update chain/table stats immediately so ETA is current. */
      pthread_mutex_lock(&stats_mutex);
      num_chains_processed += num_chains;
      num_tables_processed++;
      if (get_elapsed(&last_eta_print) >= 60.0) {
        start_timer(&last_eta_print);
        print_eta_search(num_tables_processed, wargs->total_tables);
      }
      pthread_mutex_unlock(&stats_mutex);

      flush_fa = (tables_in_scratch >= FA_BATCH_SIZE);
    }

    if (flush_fa) {
      /* Acquire a GPU device slot. */
      pthread_mutex_lock(&device_pool_mutex);
      while (device_pool_count == 0)
        pthread_cond_wait(&device_pool_cond, &device_pool_mutex);
      device_pool_count--;
      dev_slot = device_pool_slots[device_pool_count];
      pthread_mutex_unlock(&device_pool_mutex);

      /* Copy template args for this device; patch for single-device use. */
      fa_args = wargs->all_device_args[dev_slot];

      check_false_alarms_worker(wargs->ppi_head, scratch, &fa_args, dev_slot);

      /* Write back the FA kernel cache so the next use of this device slot
       * reuses the compiled kernel instead of recompiling. */
      wargs->all_device_args[dev_slot].gpu.fa_context = fa_args.gpu.fa_context;
      wargs->all_device_args[dev_slot].gpu.fa_queue   = fa_args.gpu.fa_queue;
      wargs->all_device_args[dev_slot].gpu.fa_program = fa_args.gpu.fa_program;
      wargs->all_device_args[dev_slot].gpu.fa_kernel  = fa_args.gpu.fa_kernel;

      /* Return the GPU device slot. */
      pthread_mutex_lock(&device_pool_mutex);
      device_pool_slots[device_pool_count] = dev_slot;
      device_pool_count++;
      pthread_cond_signal(&device_pool_cond);
      pthread_mutex_unlock(&device_pool_mutex);

      /* Check all_cracked after FA (which is where new cracks are registered). */
      pthread_mutex_lock(&stats_mutex);
      if (num_cracked >= num_hashes && num_hashes > 0)
        all_cracked = 1;
      pthread_mutex_unlock(&stats_mutex);

      clear_worker_scratch(scratch);
      tables_in_scratch = 0;
    }

    if (no_more_tables) {
      fprintf(stderr, "[worker %u] Exiting: no more tables.\n", wargs->worker_id);  fflush(stderr);
      break;
    }
  }

  free_worker_scratch(scratch);
  return NULL;
}


void search_tables(unsigned int total_tables, precomputed_and_potential_indices *ppi, thread_args *args) {
  unsigned int num_cores = get_num_cpu_cores();
  unsigned int num_devices = args[0].total_devices;
  unsigned int i = 0;

  /* W = number of parallel workers; at least num_devices+1 so CPU work can
   * overlap GPU work, clamped to [2, 8]. */
  unsigned int W = num_cores / 2;
  if (W < 2) W = 2;
  if (W < num_devices + 1) W = num_devices + 1;
  if (W > 8) W = 8;

  /* Split cores evenly across workers (at least 1 per worker). */
  unsigned int bs_threads = num_cores / W;
  if (bs_threads < 1) bs_threads = 1;

  /* Size the preload buffer so every worker can have FA_BATCH_SIZE tables in
   * its scratch plus one spare per worker.  W+1 was too small: with 8 workers
   * and FA_BATCH_SIZE=4, workers stall waiting for the preloader before they
   * can fill a full batch. */
  max_preload_num = W * FA_BATCH_SIZE + 1;
  /* Wake the preloader in case it's already blocked on the old (smaller) limit. */
  pthread_mutex_lock(&preloaded_tables_lock);
  pthread_cond_signal(&condition_continue_loading_tables);
  pthread_mutex_unlock(&preloaded_tables_lock);

  /* Populate the device pool with all device indices. */
  pthread_mutex_lock(&device_pool_mutex);
  for (i = 0; i < num_devices; i++)
    device_pool_slots[i] = i;
  device_pool_count = num_devices;
  pthread_mutex_unlock(&device_pool_mutex);

  printf("Binary searching with %u workers, %u threads each (total tables: %u).\n", W, bs_threads, total_tables);  fflush(stdout);

  worker_thread_args *wargs = calloc(W, sizeof(worker_thread_args));
  pthread_t *worker_threads = calloc(W, sizeof(pthread_t));
  if ((wargs == NULL) || (worker_threads == NULL)) {
    fprintf(stderr, "Failed to allocate worker args.\n"); exit(-1);
  }

  start_timer(&last_eta_print);

  for (i = 0; i < W; i++) {
    wargs[i].ppi_head = ppi;
    wargs[i].all_device_args = args;
    wargs[i].num_devices = num_devices;
    wargs[i].bs_threads = bs_threads;
    wargs[i].worker_id = i;
    wargs[i].total_tables = total_tables;

    if (pthread_create(&worker_threads[i], NULL, table_worker_thread, &wargs[i])) {
      perror("Failed to create worker thread"); exit(-1);
    }
  }

  for (i = 0; i < W; i++) {
    if (pthread_join(worker_threads[i], NULL) != 0) {
      perror("Failed to join worker thread"); exit(-1);
    }
  }

  FREE(wargs);
  FREE(worker_threads);

  /* Signal the preloader to stop (in case of early exit) and drain remaining tables. */
  pthread_mutex_lock(&preloaded_tables_lock);
  stop_preloading = 1;
  pthread_cond_broadcast(&condition_continue_loading_tables);
  while (preloaded_table_list != NULL) {
    preloaded_table *pt_next = preloaded_table_list->next;
    FREE(preloaded_table_list->filepath);
    FREE(preloaded_table_list->rainbow_table);
    FREE(preloaded_table_list);
    preloaded_table_list = pt_next;
  }
  pthread_mutex_unlock(&preloaded_tables_lock);
}


int main(int ac, char **av) {
  char *rt_dir = NULL, *single_hash = NULL, *filename = NULL, *file_data = NULL, **usernames = NULL, **hashes = NULL, *line = NULL, *pot_file_data = NULL;
  unsigned int i = 0, j = 0, max_num_hashes = 0, num_colons = 0, file_format = 0, err = 0;
  FILE *f = NULL;
  struct stat st = {0};
  thread_args *args = NULL;
  char time_precomp_str[64] = {0}, time_io_str[64] = {0}, time_searching_str[64] = {0}, time_falsealarms_str[64] = {0}, time_total_str[64] = {0}, time_per_table_str[64] = {0};

  rt_parameters rt_params = {0};

  cl_platform_id platforms[MAX_NUM_PLATFORMS] = {0};
  cl_device_id devices[MAX_NUM_DEVICES] = {0};

  cl_uint num_platforms = 0, num_devices = 0;

  precomputed_and_potential_indices *ppi_head = NULL, *ppi_cur = NULL;

  pthread_t preload_thread_id = {0};
  preloading_thread_args preload_thread_args = {0};
  ntlmv1_capture parsed_capture = {0};
  ntlmv1_capture *captures = NULL;
  unsigned int num_captures = 0, num_skipped = 0;


  ENABLE_CONSOLE_COLOR();
  PRINT_PROJECT_HEADER();
  setlocale(LC_NUMERIC, "");
  /* Scan for flag-style arguments: -ntlmv1, -gws, -disable-platform. */
  for (i = 2; (int)i < ac; i++) {
    if (strcmp(av[i], "-ntlmv1") == 0) {
      ntlmv1_mode = 1;
      if ((int)(i + 1) < ac)
        ntlmv1_capture_arg = av[++i];
    } else if (strcmp(av[i], "-ntlmv1-file") == 0) {
      ntlmv1_mode = 1;
      ntlmv1_batch_mode = 1;
      if ((int)(i + 1) < ac)
        ntlmv1_file_arg = av[++i];
    } else if (strcmp(av[i], "-gws") == 0) {
      if ((int)(i + 1) < ac)
        user_provided_gws = (unsigned int)atoi(av[++i]);
    } else if (strcmp(av[i], "-disable-platform") == 0) {
      if ((int)(i + 1) < ac)
        disable_platform = (unsigned int)atoi(av[++i]);
    }
  }

  if (ntlmv1_mode) {
    if (ntlmv1_batch_mode) {
      if (ac < 4 || ntlmv1_file_arg == NULL)
        print_usage_and_exit(av[0], -1);
    } else {
      if (ac < 4 || ntlmv1_capture_arg == NULL)
        print_usage_and_exit(av[0], -1);
    }
  } else {
    if ((ac < 3) || (ac > 5))
      print_usage_and_exit(av[0], -1);
    else if ((ac == 5) && (strcmp(av[3], "-gws") != 0) && (strcmp(av[3], "-disable-platform") != 0))
      print_usage_and_exit(av[0], -1);
  }

  /* In ntlmv1 mode, validate the capture before any GPU work. */
  if (ntlmv1_mode && !ntlmv1_batch_mode) {
    int parse_ret = ntlmv1_parse_capture(ntlmv1_capture_arg, &parsed_capture);
    if (parse_ret == NTLMV1_ERR_FORMAT) {
      fprintf(stderr, "Error: invalid NTLMv1 capture format.\n  Expected: user::domain:LMresp48hex:NTresp48hex:challenge16hex\n");
      exit(-1);
    } else if (parse_ret == NTLMV1_ERR_CHALLENGE) {
      fprintf(stderr, "Error: challenge is not 1122334455667788.\n  NTLMv1 rainbow tables require the fixed challenge; this capture cannot be cracked with tables.\n");
      exit(1);
    } else if (parse_ret == NTLMV1_ERR_ESS) {
      fprintf(stderr, "Error: ESS (Extended Session Security / NTLMv1-SSP) detected.\n  The LM response indicates a non-fixed challenge; this capture cannot be cracked with rainbow tables.\n");
      exit(1);
    }
  }

  /* Initialize the devices. */
  get_platforms_and_devices(disable_platform, MAX_NUM_PLATFORMS, platforms, &num_platforms, MAX_NUM_DEVICES, devices, &num_devices, VERBOSE);

  /* Check the device type and set flags.*/
  if (num_devices > 0) {
    char device_vendor[128] = {0};

    get_device_str(devices[0], CL_DEVICE_VENDOR, device_vendor, sizeof(device_vendor) - 1);
    if (strstr(device_vendor, "Advanced Micro Devices") != NULL)
      is_amd_gpu = 1;
  }

  /* Print a warning on Windows 7 systems, as they are observed to be highly
   * unstable for performing lookups on. */
  PRINT_WIN7_LOOKUP_WARNING();

  /* Check that this system has sufficient RAM. */
  CHECK_MEMORY_SIZE();

  /* Initialize the barrier.  This is used in some cases to ensure kernels across
   * multiple devices run concurrently. */
  if (pthread_barrier_init(&barrier, NULL, num_devices) != 0) {
    fprintf(stderr, "pthread_barrier_init() failed.\n");
    exit(-1);
  }

  printf("GPU devices: %u.  Binary search workers will be determined automatically.\n", num_devices);

  /* First arg is the directory (and/or sub-directories) containing rainbow tables. */
  rt_dir = av[1];

  /* The default rainbowcrackalack.pot file can be overridden with a third argument.
   * This is undocumented since its probably only useful for automated testing. */
  if (!ntlmv1_mode && ac == 4) {
    strncpy(jtr_pot_filename, av[3], sizeof(jtr_pot_filename) - 1);
    jtr_pot_filename[sizeof(jtr_pot_filename) - 1] = '\0';
    strncpy(hashcat_pot_filename, av[3], sizeof(hashcat_pot_filename) - 1);
    hashcat_pot_filename[sizeof(hashcat_pot_filename) - 1] = '\0';
    strncat(hashcat_pot_filename, ".hashcat", sizeof(hashcat_pot_filename) - 1);
  }

  /* Open the JTR pot file for reading.  We will check the hash(es) to see if any are
   * already cracked. */
  f = fopen(jtr_pot_filename, "rb");
  if (f) {
    unsigned long file_size = get_file_size(f);

    pot_file_data = calloc(file_size, sizeof(char));
    if (pot_file_data == NULL) {
      fprintf(stderr, "Failed to allocate buffer for pot file.\n");
      exit(-1);
    }

    if (fread(pot_file_data, sizeof(char), file_size, f) != file_size) {
      fprintf(stderr, "Error reading pot file: %s\n", strerror(errno));
      exit(-1);
    }
  } else {
    /* Allocate an empty string. */
    pot_file_data = calloc(1, sizeof(char));
    if (pot_file_data == NULL) {
      fprintf(stderr, "Failed to allocate buffer for pot file.\n");
      exit(-1);
    }
  }

  FCLOSE(f);

  /* Check if the second arg is a hash or a file containing hashes.
   * Skipped in ntlmv1 mode where av[2] is the -ntlmv1 flag, not a hash. */
  if (!ntlmv1_mode) {
  if (stat(av[2], &st) == 0)
    filename = av[2];
  else {
    single_hash = av[2];

    /* Ensure that hash is lowercase. */
    str_to_lowercase(single_hash);

    /* If this hash is already in the pot file, then there's nothing else to do. */
    if (pot_file_data && strstr(pot_file_data, single_hash)) {
      printf("Specified hash has already been cracked!  Check %s.\n", jtr_pot_filename);
      exit(0);
    }
  }

  if (filename) {
    FILE *f = fopen(filename, "rb");
    unsigned int previously_cracked = 0;


    if (f == NULL) {
      fprintf(stderr, "Error while opening file %s for reading: %s\n", filename, strerror(errno));
      goto err;
    }

    file_data = calloc(st.st_size + 1, sizeof(char));
    if (file_data == NULL) {
      fprintf(stderr, "Error while allocating buffer for hash file.\n");
      goto err;
    }

    if (fread(file_data, sizeof(char), st.st_size, f) != st.st_size) {
      fprintf(stderr, "Error while reading hash file: %s\n", strerror(errno));
      goto err;
    }

    FCLOSE(f);

    /* Count the number of newlines in the file so we know how large to make the
     * hash array. */
    for (i = 0; i < st.st_size; i++) {
      if (file_data[i] == '\n')
	max_num_hashes++;
    }
    max_num_hashes++;  /* In case the last line doesn't end with an LF. */

    num_colons = 0;
    for (i = 0; i < st.st_size; i++) {
      if (file_data[i] == ':')
        num_colons++;
      else if (file_data[i] == '\n')
        break;
    }

    if (num_colons == 0) {
      file_format = HASH_FILE_FORMAT_PLAIN;
      printf("Hash file contains plain hashes.\n");
    } else if (num_colons == 6) {
      file_format = HASH_FILE_FORMAT_PWDUMP;
      printf("Hash file is pwdump format.\n");
    } else {
      fprintf(stderr, "Error: hash file format is not recognized (number of colons in first line is %u, instead of 0 or 6).\n", num_colons);
      goto err;
    }

    usernames = calloc(max_num_hashes, sizeof(char *));
    hashes = calloc(max_num_hashes, sizeof(char *));
    if ((usernames == NULL) || (hashes == NULL)) {
      fprintf(stderr, "Error while allocating buffer for hashes.\n");
      goto err;
    }

    /* Tokenize the hash file by line.  Store each hash in the array. */
    num_hashes = 0;
    line = strtok(file_data, "\n");
    while (line && (num_hashes < max_num_hashes)) {

      /* Skip empty lines.  */
      if (strlen(line) > 0) {

	/* Skip previously-cracked hashes. */
	if (strstr(pot_file_data, line) != NULL)
	  previously_cracked++;
	else {
          /* If we're dealing with CRLF line endings, cut off the trailing CR. */
          if (line[strlen(line) - 1] == '\r')
            line[strlen(line) - 1] = '\0';

          if (file_format == HASH_FILE_FORMAT_PLAIN) {
            /* Ensure that hash is lowercase. */
            str_to_lowercase(line);

            hashes[num_hashes] = strdup(line);
            if (hashes[num_hashes] == NULL) {
              fprintf(stderr, "Error while allocating buffer for hashes.\n");
              goto err;
            }
            num_hashes++;
          } else {  /* HASH_FILE_FORMAT_PWDUMP */
            char *line_copy = strdup(line);
            char *hash = NULL;
            unsigned int line_copy_len = strlen(line_copy);
            unsigned int hash_start = 0, hash_end = 0;


            /* Get the username from position zero until the first colon. */
            for (i = 0; i < line_copy_len; i++) {
              if (line_copy[i] == ':') {
                line_copy[i] = '\0';
                usernames[num_hashes] = strdup(line_copy);
                if (usernames[num_hashes] == NULL) {
                  fprintf(stderr, "Error while allocating buffer for usernames.\n");
                  goto err;
                }
                break;
              }
            }

            /* Find the start and end positions of the hash, based on the number of colons. */
            num_colons = 1;
            hash_start = 0;
            hash_end = 0;
            for (i = i + 1; i < line_copy_len; i++) {
              if (line_copy[i] == ':')
                num_colons++;

              if ((num_colons == 3) && (hash_start == 0))
                hash_start = i + 1;
              else if (num_colons == 4) {
                hash_end = i;
                break;
              }
            }

            if ((hash_start == 0) || (hash_end == 0)) {
              fprintf(stderr, "Error: failed to extract hash from line: [%s]\n", line);
              goto err;
            }

            *(line_copy + hash_end) = '\0';
            hash = line_copy + hash_start;
            /*printf("Found hash at %u:%u: [%s]\n", hash_start, hash_end, hash);*/

            /* Make sure the hash is 32 bytes. */
            if (strlen(hash) != 32) {
              fprintf(stderr, "Error: hash is length %u instead of 32: [%s]\n", (unsigned int)strlen(hash), hash);
              goto err;
            }

            str_to_lowercase(hash);  /* Ensure hash is lowercase. */

            if (strstr(pot_file_data, hash) != NULL) {
              previously_cracked++;
            } else {
              hashes[num_hashes] = strdup(hash);
              if (hashes[num_hashes] == NULL) {
                fprintf(stderr, "Error while allocating buffer for hashes.\n");
                goto err;
              }
              num_hashes++;
            }
            FREE(line_copy);

          }
        }
	line = strtok(NULL, "\n");
      }
    }

    FREE(file_data);

    if (num_hashes == 0) {
      printf("All hashes have already been cracked!  Check %s.\n", jtr_pot_filename);
      exit(0);
    } else {
      printf("Loaded %u of %u uncracked hashes from %s.\n", num_hashes, num_hashes + previously_cracked, filename);  fflush(stdout);
    }

  } else { /* A single hash was provided. */
    usernames = calloc(1, sizeof(char *));
    hashes = calloc(1, sizeof(char *));
    if ((usernames == NULL) || (hashes == NULL)) {
      fprintf(stderr, "Error while allocating buffer for hashes.\n");
      goto err;
    }

    usernames[0] = NULL;
    hashes[0] = strdup(single_hash);
    num_hashes = 1;
  }
  } else { /* ntlmv1 mode: inject block hashes */
    if (!ntlmv1_batch_mode) {
      /* Single capture: inject two block hashes directly. */
      usernames = calloc(2, sizeof(char *));
      hashes = calloc(2, sizeof(char *));
      if ((usernames == NULL) || (hashes == NULL)) {
        fprintf(stderr, "Error while allocating buffer for hashes.\n");
        goto err;
      }
      usernames[0] = usernames[1] = NULL;
      hashes[0] = strdup(parsed_capture.block1_hex);
      hashes[1] = strdup(parsed_capture.block2_hex);
      num_hashes = 2;
    } else {
      /* Batch mode: read captures file, parse lines, dedup-insert block hashes. */
      FILE *batch_f = NULL;
      char *batch_file_data = NULL;
      unsigned long batch_file_size = 0;
      unsigned long k = 0;
      char *line = NULL;
      unsigned int max_lines = 0, h = 0, ln = 0;

      batch_f = fopen(ntlmv1_file_arg, "rb");
      if (batch_f == NULL) {
        fprintf(stderr, "Error: could not open captures file: %s\n", ntlmv1_file_arg);
        goto err;
      }
      batch_file_size = get_file_size(batch_f);
      if (batch_file_size == 0) {
        fprintf(stderr, "Error: captures file is empty: %s\n", ntlmv1_file_arg);
        fclose(batch_f);
        goto err;
      }
      batch_file_data = calloc(batch_file_size + 1, 1);
      if (batch_file_data == NULL) {
        fprintf(stderr, "Error: could not allocate buffer for captures file.\n");
        fclose(batch_f);
        goto err;
      }
      if (fread(batch_file_data, 1, batch_file_size, batch_f) != batch_file_size) {
        fprintf(stderr, "Error: could not read captures file: %s\n", ntlmv1_file_arg);
        fclose(batch_f);
        FREE(batch_file_data);
        goto err;
      }
      fclose(batch_f);
      batch_file_data[batch_file_size] = '\0';

      /* Count newlines to bound the array sizes. */
      for (k = 0; k < batch_file_size; k++) {
        if (batch_file_data[k] == '\n')
          max_lines++;
      }
      max_lines++;  /* account for final line with no trailing newline */

      captures = calloc(max_lines, sizeof(ntlmv1_capture));
      usernames = calloc(max_lines * 2, sizeof(char *));
      hashes = calloc(max_lines * 2, sizeof(char *));
      if ((captures == NULL) || (usernames == NULL) || (hashes == NULL)) {
        fprintf(stderr, "Error while allocating buffer for batch captures.\n");
        FREE(batch_file_data);
        goto err;
      }

      /* Parse each line and dedup-insert block hashes. */
      line = strtok(batch_file_data, "\n");
      while (line != NULL) {
        ntlmv1_capture tmp = {0};
        unsigned int llen = (unsigned int)strlen(line);
        int ret = 0;

        ln++;

        /* Trim trailing carriage return. */
        if (llen > 0 && line[llen - 1] == '\r')
          line[--llen] = '\0';

        if (llen == 0) {
          line = strtok(NULL, "\n");
          continue;
        }

        ret = ntlmv1_parse_capture(line, &tmp);
        if (ret != NTLMV1_OK) {
          if (ret == NTLMV1_ERR_ESS)
            printf("  Line %u: skipped (ESS/NTLMv1-SSP): %s\n", ln, line);
          else if (ret == NTLMV1_ERR_CHALLENGE)
            printf("  Line %u: skipped (challenge != 1122334455667788): %s\n", ln, line);
          else
            printf("  Line %u: skipped (bad format): %s\n", ln, line);
          num_skipped++;
          line = strtok(NULL, "\n");
          continue;
        }

        captures[num_captures++] = tmp;  /* struct copy */

        /* Dedup-insert block1_hex. */
        for (h = 0; h < num_hashes; h++) {
          if (strcmp(hashes[h], tmp.block1_hex) == 0)
            break;
        }
        if (h == num_hashes) {
          hashes[num_hashes] = strdup(tmp.block1_hex);
          usernames[num_hashes] = NULL;
          num_hashes++;
        }

        /* Dedup-insert block2_hex. */
        for (h = 0; h < num_hashes; h++) {
          if (strcmp(hashes[h], tmp.block2_hex) == 0)
            break;
        }
        if (h == num_hashes) {
          hashes[num_hashes] = strdup(tmp.block2_hex);
          usernames[num_hashes] = NULL;
          num_hashes++;
        }

        line = strtok(NULL, "\n");
      }

      FREE(batch_file_data);

      if (num_captures == 0) {
        printf("No valid NTLMv1 captures found in %s (%u line(s) skipped).\n", ntlmv1_file_arg, num_skipped);
        free_loaded_hashes(usernames, hashes);
        FREE(captures);
        FREE(args);
        pthread_barrier_destroy(&barrier);
        return 0;
      }

      printf("Loaded %u capture(s) from %s (%u skipped); %u unique block hash(es) to precompute.\n",
             num_captures, ntlmv1_file_arg, num_skipped, num_hashes);
      fflush(stdout);
    }
  }

  /* We're done checking the pot file for previously-cracked hashes. */
  FREE(pot_file_data);

  /* Look through the supplied rainbow table directory, and infer the parameters via
   * the filenames. */
  find_rt_params(rt_dir, &rt_params);
  if (!rt_params.parsed) {
    fprintf(stderr, "Failed to infer rainbow table parameters from files in directory.  Ensure that valid rainbow table files are in %s (and/or its sub-directories).\n", rt_dir);
    exit(-1);
  }

  /* At this time, only NTLM hashes are supported.
  if (rt_params.hash_type != HASH_NTLM) {
    fprintf(stderr, "Unfortunately, only NTLM hashes are supported at this time.  Terminating.\n");
    exit(-1);
  }
  */

  if (ntlmv1_mode && rt_params.hash_type != HASH_NETNTLMV1) {
    fprintf(stderr, "Error: -ntlmv1 requires netntlmv1 rainbow tables (found hash type %u in %s).\n", rt_params.hash_type, rt_dir);
    exit(-1);
  }

  /* Ensure that valid hashes were provided. */
  if (rt_params.hash_type == HASH_NTLM) {
    for (i = 0; i < num_hashes; i++) {
      if (strlen(hashes[i]) != 32) {
	fprintf(stderr, "Error: invalid NTLM hash (length is not 32!): %s\n", hashes[i]);
	exit(-1);
      }
    }
  }

  /* Issue a warning if more than 5,000 hashes were provided, as rainbow tables may
   * start to become not as efficient as brute-force. */
  if (num_hashes > 5000) {
    printf("\n\n\n\t!! WARNING !!\n\nA large group of hashes was provided (%u).  In general, rainbow tables are only effective to use for small numbers of hashes because there is a pre-computation step that must be done on *each hash*; eventually this pre-computation cost becomes high enough that brute-force would be a better strategy.  The point at which this happens depends on your specific GPU hardware.\n\nFor example, suppose the pre-computation step takes 2.8 seconds per hash, and brute-forcing takes 16 hours (57,600 seconds).  Not counting search time nor false alarm checking, the point at which brute-forcing becomes more efficient than rainbow tables is: 57,600 / 2.8 = ~20,571 hashes.  Trying to crack more than this number of hashes is clearly less effective than brute-force.\n\nPay attention to the pre-computation times below, and compare with the reported estimate that hashcat gives after a few minutes for brute-forcing 8-character NTLM (hint: ./hashcat -m 1000 -a 3 -w 3 -O ffffffffffffffffffffffffffffffff ?a?a?a?a?a?a?a?a).\n\n\n\n", num_hashes);  fflush(stdout);
  }

  args = calloc(num_devices, sizeof(thread_args));
  if (args == NULL) {
    fprintf(stderr, "Error while creating thread arg array.\n");
    goto err;
  }

  /* We set most of the args once, since all GPUs & hashes need all the same
   * parameters. */
  for (i = 0; i < num_devices; i++) {
    args[i].hash_type = rt_params.hash_type;
    args[i].hash_name = rt_params.hash_name;
    args[i].username = NULL;  /* Filled in below. */
    args[i].hash = NULL;      /* Filled in below. */
    args[i].charset = validate_charset(rt_params.charset_name);
    args[i].charset_name = rt_params.charset_name;
    args[i].plaintext_len_min = rt_params.plaintext_len_min;
    args[i].plaintext_len_max = rt_params.plaintext_len_max;
    args[i].table_index = rt_params.table_index;
    args[i].reduction_offset = rt_params.reduction_offset;
    args[i].chain_len = rt_params.chain_len;
    args[i].total_devices = num_devices;
    args[i].gpu.device_number = i;
    args[i].gpu.device = devices[i];
    get_device_uint(args[i].gpu.device, CL_DEVICE_MAX_COMPUTE_UNITS, &(args[i].gpu.num_work_units));
  }

  /* Autotune GWS for each device empirically unless the user specified one. */
  if (user_provided_gws == 0) {
    for (i = 0; i < num_devices; i++) {
      args[i].hash = hashes[0];
      args[i].gpu.tuned_gws = autotune_precompute_gws(&args[i]);
      args[i].hash = NULL;
    }
  }

  /* Pre-warm the FA kernel on every device to JIT-compile it now, eliminating
   * the ~11.5 s compilation delay that would otherwise hit the first table. */
  for (i = 0; i < num_devices; i++) {
    prewarm_fa_kernel(&args[i]);
  }

  num_hashes_precomputed_total = num_hashes;
  start_timer(&precompute_start_time);
  for (i = 0; i < num_hashes; i++) {
    printf("Pre-computing hash #%u: %s...\n", i + 1, hashes[i]);  fflush(stdout);

    for (j = 0; j < num_devices; j++) {
      args[j].username = usernames[i];
      args[j].hash = hashes[i];
    }

    precompute_hash(num_devices, args, &ppi_head);
  }
  time_precomp = get_elapsed(&precompute_start_time);
  seconds_to_human_time(time_precomp_str, sizeof(time_precomp_str), time_precomp);
  printf("\nPre-computation finished in %s.\n\n", time_precomp_str);  fflush(stdout);

  /* If too much memory is taken up by the pre-computed indices, print a warning to the
   * user.  Strange crashes in the OpenCL functions can occur when memory is exhausted,
   * and its not obvious that this is the culprit. */
  check_memory_usage();

  /* Set max_preload_num before starting the preloader so it doesn't block after
   * only 2 tables.  search_tables() will recalculate with the exact W, but this
   * estimate prevents unnecessary early stalling. */
  {
    unsigned int est_cores = get_num_cpu_cores();
    unsigned int est_W = est_cores / 2;
    if (est_W < 2) est_W = 2;
    if (est_W > 8) est_W = 8;
    max_preload_num = est_W * FA_BATCH_SIZE + 1;
  }

  /* Start preloading tables into memory. */
  preload_thread_args.rt_dir = strdup(rt_dir);
  err = pthread_create(&preload_thread_id, NULL, preloading_thread, &preload_thread_args);
  if (err != 0) {
    printf("Failed to create thread: %d\n", err);
    return -1;
  }

  /* Using the pre-computed end indices, perform a binary search on all rainbow tables
   * in the target directory.  Any matching indices will trigger false alarm checks. */
  total_tables = count_tables(rt_dir);
  start_timer(&search_start_time);
  search_tables(total_tables, ppi_head, args);

  /* Join the preload thread (it may still be running if we exited early). */
  pthread_join(preload_thread_id, NULL);

  /* NTLMv1 reassembly: match recovered block keys and assemble the full NT hash. */
  if (ntlmv1_mode) {
    if (!ntlmv1_batch_mode) {
      /* Single capture reassembly. */
      unsigned char k1[7] = {0}, k2[7] = {0};
      int k1_found = 0, k2_found = 0;
      int same_block = (strcmp(parsed_capture.block1_hex, parsed_capture.block2_hex) == 0);

      ppi_cur = ppi_head;
      while (ppi_cur != NULL) {
        if (ppi_cur->cracked_key_len > 0) {
          if (strcmp(ppi_cur->hash, parsed_capture.block1_hex) == 0) {
            memcpy(k1, ppi_cur->cracked_key, 7);
            k1_found = 1;
            if (same_block) {
              memcpy(k2, ppi_cur->cracked_key, 7);
              k2_found = 1;
            }
          }
          if (!same_block && strcmp(ppi_cur->hash, parsed_capture.block2_hex) == 0) {
            memcpy(k2, ppi_cur->cracked_key, 7);
            k2_found = 1;
          }
        }
        ppi_cur = ppi_cur->next;
      }

      if (!k1_found || !k2_found) {
        printf("\nPartial NTLMv1 crack:\n");
        if (k1_found) {
          char hex[15] = {0};
          bytes_to_hex(k1, 7, hex, sizeof(hex));
          printf("  Block 1 (%s): %s\n", parsed_capture.block1_hex, hex);
        } else {
          printf("  Block 1 (%s): NOT FOUND in tables\n", parsed_capture.block1_hex);
        }
        if (k2_found) {
          char hex[15] = {0};
          bytes_to_hex(k2, 7, hex, sizeof(hex));
          printf("  Block 2 (%s): %s\n", parsed_capture.block2_hex, hex);
        } else {
          printf("  Block 2 (%s): NOT FOUND in tables\n", parsed_capture.block2_hex);
        }
        printf("  Full NT hash cannot be assembled.\n");
      } else {
        unsigned char last2[2] = {0};
        unsigned int found_count = 0;
        int ok = ntlmv1_recover_last2(parsed_capture.block3, last2, &found_count);
        if (!ok || found_count == 0) {
          printf("\nCould not brute-force block3; capture may be malformed.\n");
        } else {
          unsigned char ntlm16[16] = {0};
          char ntlm_hex[33] = {0};

          if (found_count > 1)
            printf("\nWarning: block3 brute-force produced %u collisions; using first match.\n", found_count);

          ntlmv1_assemble(k1, k2, last2, ntlm16);
          bytes_to_hex(ntlm16, 16, ntlm_hex, sizeof(ntlm_hex));

          printf("\n%sRecovered NT hash: %s%s\n", GREENB, ntlm_hex, CLR);
          printf("%s%s:::%s:::%s\n", GREENB, parsed_capture.username, ntlm_hex, CLR);
          fflush(stdout);
        }
      }
    } else {
      /* Batch mode: iterate over all captures. */
      unsigned int batch_cracked = 0, batch_partial = 0, batch_failed = 0;
      unsigned int c = 0;

      for (c = 0; c < num_captures; c++) {
        unsigned char k1[7] = {0}, k2[7] = {0};
        int k1_found = 0, k2_found = 0;
        int same_block = (strcmp(captures[c].block1_hex, captures[c].block2_hex) == 0);

        ppi_cur = ppi_head;
        while (ppi_cur != NULL) {
          if (ppi_cur->cracked_key_len > 0) {
            if (strcmp(ppi_cur->hash, captures[c].block1_hex) == 0) {
              memcpy(k1, ppi_cur->cracked_key, 7);
              k1_found = 1;
              if (same_block) {
                memcpy(k2, ppi_cur->cracked_key, 7);
                k2_found = 1;
              }
            }
            if (!same_block && strcmp(ppi_cur->hash, captures[c].block2_hex) == 0) {
              memcpy(k2, ppi_cur->cracked_key, 7);
              k2_found = 1;
            }
          }
          ppi_cur = ppi_cur->next;
        }

        if (!k1_found || !k2_found) {
          printf("\n[%s] Partial NTLMv1 crack:\n", captures[c].username);
          if (k1_found) {
            char hex[15] = {0};
            bytes_to_hex(k1, 7, hex, sizeof(hex));
            printf("  Block 1 (%s): %s\n", captures[c].block1_hex, hex);
          } else {
            printf("  Block 1 (%s): NOT FOUND in tables\n", captures[c].block1_hex);
          }
          if (k2_found) {
            char hex[15] = {0};
            bytes_to_hex(k2, 7, hex, sizeof(hex));
            printf("  Block 2 (%s): %s\n", captures[c].block2_hex, hex);
          } else {
            printf("  Block 2 (%s): NOT FOUND in tables\n", captures[c].block2_hex);
          }
          printf("  Full NT hash cannot be assembled.\n");
          batch_partial++;
        } else {
          unsigned char last2[2] = {0};
          unsigned int found_count = 0;
          int ok = ntlmv1_recover_last2(captures[c].block3, last2, &found_count);
          if (!ok || found_count == 0) {
            printf("\n[%s] Could not brute-force block3; capture may be malformed.\n", captures[c].username);
            batch_failed++;
          } else {
            unsigned char ntlm16[16] = {0};
            char ntlm_hex[33] = {0};

            if (found_count > 1)
              printf("\nWarning: block3 brute-force produced %u collisions; using first match.\n", found_count);

            ntlmv1_assemble(k1, k2, last2, ntlm16);
            bytes_to_hex(ntlm16, 16, ntlm_hex, sizeof(ntlm_hex));

            printf("\n%sRecovered NT hash: %s%s\n", GREENB, ntlm_hex, CLR);
            printf("%s%s:::%s:::%s\n", GREENB, captures[c].username, ntlm_hex, CLR);
            fflush(stdout);
            batch_cracked++;
          }
        }
      }

      printf("\nNTLMv1 batch summary: %u cracked, %u partial, %u malformed-block3, %u skipped (of %u lines).\n",
             batch_cracked, batch_partial, batch_failed, num_skipped, num_captures + num_skipped);
    }

    free_precomputed_and_potential_indices(&ppi_head);
    free_loaded_hashes(usernames, hashes);
    FREE(captures);
    FREE(args);
    pthread_barrier_destroy(&barrier);
    return 0;
  }

  seconds_to_human_time(time_precomp_str, sizeof(time_precomp_str), time_precomp);
  seconds_to_human_time(time_io_str, sizeof(time_io_str), time_io);
  seconds_to_human_time(time_searching_str, sizeof(time_searching_str), time_searching);
  seconds_to_human_time(time_falsealarms_str, sizeof(time_falsealarms_str), time_falsealarms);
  seconds_to_human_time(time_total_str, sizeof(time_total_str), time_precomp + /*time_io +*/ time_searching + time_falsealarms);
  seconds_to_human_time(time_per_table_str, sizeof(time_per_table_str), (double)(time_precomp + time_io + time_searching + time_falsealarms) / (double)num_tables_processed);

  printf("\n\n        %sRAINBOW CRACKALACK LOOKUP REPORT%s\n\n", WHITEB, CLR);

  if (num_cracked == 0)
    printf("\nNo hashes were cracked.  :(\n\n\n");
  else {
    printf(" %s* Crack Summary *%s\n\n", WHITEB, CLR);
    printf("   Of the %u hashes loaded, %u were cracked, or %.2f%%.\n\n", num_hashes, num_cracked, ((double)num_cracked / (double)num_hashes) * 100);

    printf(" Results\n -------\n%s", GREENB);
    ppi_cur = ppi_head;
    while(ppi_cur != NULL) {
      if (ppi_cur->plaintext != NULL) {
        if (sizeof(ppi_cur->hash) == 8) {
          char ptxt_hex[(sizeof(ppi_cur->plaintext) * 2) + 1] = {0};
          bytes_to_hex((unsigned char*)ppi_cur->plaintext, 7, ptxt_hex, sizeof(ptxt_hex));
	  printf(" %s  %s\n", (ppi_cur->username != NULL) ? ppi_cur->username : ppi_cur->hash, ptxt_hex);

        }
        else {
	  printf(" %s  %s\n", (ppi_cur->username != NULL) ? ppi_cur->username : ppi_cur->hash, ppi_cur->plaintext);
        }
      }

      ppi_cur = ppi_cur->next;
    }
    printf("%s -------\n\n", CLR);
    printf("%s Results have been written in JTR format to:     %s\n", WHITEB, jtr_pot_filename);
    printf(" Results have been written in hashcat format to: %s%s\n\n\n", hashcat_pot_filename, CLR);
  }

  printf(" %s* Time Summary *%s\n\n      Precomputation: %s\n      I/O (parallel): %s\n           Searching: %s\n  False alarm checks: %s\n\n               Total: %s\n\n\n", WHITEB, CLR, time_precomp_str, time_io_str, time_searching_str, time_falsealarms_str, time_total_str);

  printf(" %s* Statistics *%s\n\n          Number of tables processed: %u\n              Number of false alarms: %" QUOTE PRIu64"\n          Number of chains processed: %" QUOTE PRIu64"\n\n                Time spent per table: %s\n     False alarms checked per second: %" QUOTE ".1f\n\n         False alarms per no. chains: %.5f%%\n  Successful cracks per false alarms: %.5f%%\n  Successful cracks per total chains: %.8f%%\n\n\n", WHITEB, CLR, num_tables_processed, num_falsealarms, num_chains_processed, time_per_table_str, (double)num_falsealarms / time_falsealarms, ((double)num_falsealarms / (double)num_chains_processed) * 100.0, ((double)num_cracked / (double)num_falsealarms) * 100.0, ((double)num_cracked / (double)num_chains_processed) * 100.0);

  free_precomputed_and_potential_indices(&ppi_head);
  free_loaded_hashes(usernames, hashes);
  FREE(args);
  pthread_barrier_destroy(&barrier);
  return 0;

 err:
  FCLOSE(f);
  FREE(file_data);
  free_precomputed_and_potential_indices(&ppi_head);
  free_loaded_hashes(usernames, hashes);
  FREE(captures);
  FREE(args);
  pthread_barrier_destroy(&barrier);
  return -1;
}
