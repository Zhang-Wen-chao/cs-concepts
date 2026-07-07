CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -g

BRPC_PATH ?= /opt/homebrew
PROTOBUF_PATH ?= /opt/homebrew/opt/protobuf@29
ABSL_PATH ?= /opt/homebrew/opt/abseil
PROTOC ?= $(PROTOBUF_PATH)/bin/protoc

INCLUDES := -I$(PROTOBUF_PATH)/include -I$(BRPC_PATH)/include

ABSL_LIBS := -L$(ABSL_PATH)/lib \
        -labsl_log_severity \
        -labsl_raw_logging_internal \
        -labsl_spinlock_wait \
        -labsl_base \
        -labsl_leak_check \
        -labsl_log_internal_nullguard \
        -labsl_throw_delegate \
        -labsl_string_view \
        -labsl_strings_internal \
        -labsl_int128 \
        -labsl_strings \
        -labsl_debugging_internal \
        -labsl_malloc_internal \
        -labsl_stacktrace \
        -labsl_utf8_for_code_point \
        -labsl_decode_rust_punycode \
        -labsl_demangle_rust \
        -labsl_demangle_internal \
        -labsl_symbolize \
        -labsl_examine_stack \
        -labsl_civil_time \
        -labsl_time_zone \
        -labsl_time \
        -labsl_log_internal_globals \
        -labsl_str_format_internal \
        -labsl_log_internal_format \
        -labsl_log_internal_proto \
        -labsl_city \
        -labsl_hash \
        -labsl_log_internal_fnmatch \
        -labsl_graphcycles_internal \
        -labsl_kernel_timeout_internal \
        -labsl_tracing_internal \
        -labsl_synchronization \
        -labsl_vlog_config_internal \
        -labsl_log_globals \
        -labsl_log_entry \
        -labsl_log_sink \
        -labsl_log_internal_log_sink_set \
        -labsl_log_internal_structured_proto \
        -labsl_strerror \
        -labsl_log_internal_message \
        -labsl_log_internal_check_op \
        -labsl_log_internal_conditions \
        -labsl_crc_cpu_detect \
        -labsl_crc_internal \
        -labsl_crc32c \
        -labsl_crc_cord_state \
        -labsl_cord_internal \
        -labsl_exponential_biased \
        -labsl_cordz_functions \
        -labsl_cordz_handle \
        -labsl_cordz_info \
        -labsl_cord \
        -labsl_die_if_null \
        -labsl_flags_commandlineflag_internal \
        -labsl_flags_commandlineflag \
        -labsl_flags_program_name \
        -labsl_flags_config \
        -labsl_flags_marshalling \
        -labsl_flags_internal \
        -labsl_flags_private_handle_accessor \
        -labsl_hashtablez_sampler \
        -labsl_raw_hash_set \
        -labsl_flags_reflection \
        -labsl_log_initialize \
        -labsl_random_distributions \
        -labsl_random_internal_platform \
        -labsl_random_internal_randen_hwaes_impl \
        -labsl_random_internal_randen_hwaes \
        -labsl_random_internal_randen_slow \
        -labsl_random_internal_randen \
        -labsl_random_internal_seed_material \
        -labsl_random_seed_gen_exception \
        -labsl_random_internal_entropy_pool \
        -labsl_random_seed_sequences \
        -labsl_status \
        -labsl_statusor

PROTOBUF_LIBS := -L$(PROTOBUF_PATH)/lib -lutf8_validity -lutf8_range -lprotobuf

LIBS := $(ABSL_LIBS) $(PROTOBUF_LIBS) -L$(BRPC_PATH)/lib -lbrpc \
        -lleveldb -lgflags -lssl -lcrypto -lz -lpthread

PROTO_SRCS := recommend.pb.cc
PROTO_HDRS := recommend.pb.h

TARGETS := server client

all: $(TARGETS)

server: server.cpp $(PROTO_SRCS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@ $(LIBS)

client: client.cpp $(PROTO_SRCS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@ $(LIBS)

$(PROTO_SRCS) $(PROTO_HDRS): recommend.proto
	$(PROTOC) --cpp_out=. --proto_path=. $<

clean:
	rm -f $(TARGETS) recommend.pb.* *.o
	rm -rf *.dSYM

.PHONY: all clean
