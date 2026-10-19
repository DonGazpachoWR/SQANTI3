import pickle
import shutil
import os,copy
import re
import tempfile
import time
import psutil

from multiprocessing import Process, get_start_method
from multiprocessing.connection import wait
from Bio import SeqIO


from src.qc_computations import classify_fsm, full_length_quantification, process_rts, isoform_expression_info #type: ignore
from src.qc_pipeline import run
from src.parsers import reference_parser
from src.helpers import get_corr_filenames, get_class_junc_filenames, get_isoform_hits_name, get_pickle_filename, rename_novel_genes
from src.qc_output import (
    cleanup, generate_report, generate_tusco_report, write_classification_output, write_isoform_hits, write_junction_output, write_omitted_isoforms, write_collapsed_GFF_with_CDS)
from src.module_logging import qc_logger

# TODO: Create a special logging for the parallelization, to handle the individual logs of the splits into  their own files

def get_split_dir(outdir,prefix):
    split_prefix=os.path.join(os.path.abspath(outdir), prefix)
    split_directory = split_prefix+'_splits/'
    return split_directory

def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

# Root, limit, usage and reclaimable-cache names of each cgroup memory layout
CGROUP_MEMORY_LAYOUTS = {
    'memory': ('/sys/fs/cgroup/memory', 'memory.limit_in_bytes',
               'memory.usage_in_bytes', 'total_inactive_file'),  # cgroup v1
    '': ('/sys/fs/cgroup', 'memory.max', 'memory.current',
         'inactive_file'),  # cgroup v2
}

def read_cgroup_value(path):
    """Return the integer in a cgroup file, or None if it is missing or 'max'."""
    try:
        with open(path) as f:
            value = int(f.read().strip())
    except (OSError, ValueError):
        value = None
    return value

def read_cgroup_stat(path, key):
    """Return one field of a cgroup memory.stat file, or 0 if it is missing."""
    try:
        with open(path) as f:
            fields = dict(line.split() for line in f if len(line.split()) == 2)
    except OSError:
        fields = {}
    return int(fields.get(key, 0))

def read_cgroup_memory():
    """
    Return (used, limit) in bytes for the most restrictive memory cgroup that
    holds this process, or None if no limit is set. SLURM sets --mem on the job
    or step cgroup, not on the task cgroup of the process, so the cgroups above
    it are checked too. Inactive file cache can be reclaimed and is not counted
    as used.
    """
    result = None
    try:
        with open('/proc/self/cgroup') as f:
            cgroup_lines = f.read().splitlines()
    except OSError:
        cgroup_lines = []
    total_memory = psutil.virtual_memory().total
    for line in cgroup_lines:
        _, controllers, path = line.split(':', 2)
        controller = 'memory' if 'memory' in controllers.split(',') else controllers
        if result is None and controller in CGROUP_MEMORY_LAYOUTS:
            root, limit_name, usage_name, inactive_name = CGROUP_MEMORY_LAYOUTS[controller]
            directory = root + path.rstrip('/')
            while directory.startswith(root):
                limit = read_cgroup_value(os.path.join(directory, limit_name))
                usage = read_cgroup_value(os.path.join(directory, usage_name))
                # An unlimited cgroup reports 'max' (v2) or a huge number (v1)
                if (limit is not None and usage is not None and limit < total_memory
                        and (result is None or limit < result[1])):
                    inactive = read_cgroup_stat(os.path.join(directory, 'memory.stat'),
                                                inactive_name)
                    result = (usage - inactive, limit)
                directory = os.path.dirname(directory)
    return result

def memory_use_percent():
    """
    Percentage of memory in use: the larger of the use of the whole machine and,
    inside a cgroup with a memory limit (a SLURM job with --mem, for instance),
    the use of that cgroup against its limit.
    """
    percent = psutil.virtual_memory().percent
    cgroup_memory = read_cgroup_memory()
    if cgroup_memory is not None:
        used, limit = cgroup_memory
        percent = max(percent, round(100 * used / limit, 1))
    return percent

def read_shared_inputs(args):
    """
    Read the genome and parse the reference annotation once for all the chunks.
    Forked workers share them with this process instead of reading them again.
    With another start method they would be pickled for every worker, so None
    is returned and each worker reads its own, as before.
    """
    genome_dict, reference = None, None
    if get_start_method() == 'fork':
        qc_logger.info(f"Reading genome fasta {args.refFasta}")
        genome_dict = dict((r.name, r) for r in SeqIO.parse(open(args.refFasta), 'fasta'))
        # The genePred written by reference_parser is only needed while parsing
        with tempfile.TemporaryDirectory(dir=args.dir) as ref_dir:
            os.makedirs(os.path.join(ref_dir, 'logs'))
            reference = reference_parser(args.refGTF, ref_dir, args.output,
                                         list(genome_dict.keys()), args.genename,
                                         args.isoAnnotLite)
    return genome_dict, reference

def split_input_run(args, outdir, genome_dict=None, reference=None):
    SPLIT_ROOT_DIR = outdir
    if os.path.exists(SPLIT_ROOT_DIR):
        qc_logger.warning(f"{SPLIT_ROOT_DIR} directory already exists!")
    else:
        os.makedirs(SPLIT_ROOT_DIR)

    if not args.fasta:
        file_size = os.path.getsize(args.isoforms)
        target_chunk_bytes = max(1, file_size // args.chunks + (1 if file_size % args.chunks else 0))

        split_outs = []
        chunk_index = 0
        current_chunk_bytes = 0
        has_primary_feature = False
        has_non_comment_record = False
        
        d = os.path.join(SPLIT_ROOT_DIR, str(chunk_index))
        os.makedirs(d, exist_ok=True)
        current_f = open(os.path.join(d, os.path.basename(args.isoforms) + '.split' + str(chunk_index)), 'w')
        split_outs.append((os.path.abspath(d), current_f.name))

        primary_features = ('\ttranscript\t', '\tmRNA\t', '\tmatch\t', '\tcDNA_match\t')

        try:
            with open(args.isoforms, 'r') as f_in:
                for line in f_in:
                    if line.startswith('#'):
                        current_f.write(line)
                        continue
                    
                    has_non_comment_record = True
                    
                    # Check for boundary (a primary transcript line)
                    is_boundary = any(feat in line for feat in primary_features)
                    if is_boundary:
                        has_primary_feature = True
                    
                    # If we've reached the target chunk size, we only split AT a boundary
                    # so we don't sever a transcript from its downstream exons
                    if current_chunk_bytes >= target_chunk_bytes and is_boundary:
                        current_f.close()
                        chunk_index += 1
                        current_chunk_bytes = 0
                        
                        d = os.path.join(SPLIT_ROOT_DIR, str(chunk_index))
                        os.makedirs(d, exist_ok=True)
                        current_f = open(os.path.join(d, os.path.basename(args.isoforms) + '.split' + str(chunk_index)), 'w')
                        split_outs.append((os.path.abspath(d), current_f.name))

                    current_f.write(line)
                    current_chunk_bytes += len(line.encode('utf-8'))
        finally:
            if not current_f.closed:
                current_f.close()
        
        if not has_non_comment_record:
            qc_logger.error(f"Input file '{args.isoforms}' contains only comments or is empty.")
            raise ValueError(f"Input file '{args.isoforms}' contains only comments or is empty.")
            
        if not has_primary_feature:
            qc_logger.warning(f"No primary features (e.g., transcript, mRNA) detected in '{args.isoforms}'. SQANTI3 might fail to process this file correctly.")

    else:
        # FASTA file handling remains unchanged
        recs = [r for r in SeqIO.parse(open(args.isoforms),'fasta')]
        n = len(recs)
        chunk_size = n//args.chunks + (n%args.chunks >0)
        split_outs = []
        for i in range(args.chunks):
            if i*chunk_size >= n:
                break
            d = os.path.join(SPLIT_ROOT_DIR, str(i))
            os.makedirs(d)
            f = open(os.path.join(d, os.path.basename(args.isoforms)+'.split'+str(i)), 'w')
            for j in range(i*chunk_size, min((i+1)*chunk_size, n)):
                SeqIO.write(recs[j], f, 'fasta')
            f.close()
            split_outs.append((os.path.abspath(d), f.name))

    pools = []
    
    # -t is the CPU budget of the whole run (sequence_correction() already
    # gives each worker's aligner cpus // chunks), and only the CPUs this
    # process may use count, such as a SLURM allocation, not every CPU of the
    # machine. A worker classifies on one CPU, so up to that many run at once
    if hasattr(os, 'sched_getaffinity'):
        available_cpus = len(os.sched_getaffinity(0))
    else:
        available_cpus = psutil.cpu_count(logical=True) or 1
    max_workers = max(1, min(args.cpus, available_cpus))

    pending_tasks = list(enumerate(split_outs))
    active_processes = []

    while pending_tasks or active_processes:
        # Keep only alive processes
        active_processes = [p for p in active_processes if p.is_alive()]
        
        # Check memory, against the job's limit when there is one. We need to
        # reserve some overhead (e.g., >15% free memory) for each new chunk
        mem_percent = memory_use_percent()
        can_launch_mem = mem_percent < 85  or len(active_processes) == 0
        
        # max_workers is at least 1, so a worker can always start when none is running
        can_launch_cpu = len(active_processes) < max_workers

        # Launch next chunk if resources are safe
        if pending_tasks and can_launch_cpu and can_launch_mem:
            i, (d, x) = pending_tasks.pop(0)
            qc_logger.info(f"Launching worker on {x} (Active workers: {len(active_processes)+1}, Mem usage: {mem_percent}%)")
            
            args2 = copy.deepcopy(args)
            args2.isoforms = x
            args2.novel_gene_prefix = str(i)
            args2.dir = d
            args2.report = 'skip'
            
            p = Process(target=run, args=(args2, genome_dict, reference))
            p.start()
            
            active_processes.append(p)
            pools.append(p)
            
            # Brief pause to let memory initialize before spawning another chunk.
            # Only a worker that reads its own genome and reference needs it;
            # forked workers share those of this process
            if genome_dict is None:
                time.sleep(2)
        else:
            # Wait for resources to free up: until a worker ends, checking
            # again at least every second
            wait([p.sentinel for p in active_processes], timeout=1)

    for p in pools:
        p.join()
    return [d for (d,x) in split_outs]

def combine_split_runs(args, split_dirs, genome_dict=None):
    """
    Combine .faa, .fasta, .gtf, .classification.txt, .junctions.txt
    Then write out the PDF report
    """
    corrGTF, _, corrFASTA, corrORF , corrCDS_GTF_GFF = get_corr_filenames(args.dir, args.output)
    outputClassPath, outputJuncPath = get_class_junc_filenames(args.dir,args.output)

    if args.isoform_hits:
        isoform_hits = open(get_isoform_hits_name(args.dir, args.output)+'_tmp', 'w')
    if args.include_ORF:
        f_faa = open(corrORF, 'w')
    f_fasta = open(corrFASTA, 'w')
    f_gtf = open(corrGTF, 'w')
    f_junc_temp = open(outputJuncPath+"_tmp", "w")

    isoforms_info = {}
    headers = []
    for i,split_d in enumerate(split_dirs):
        _gtf, _, _fasta, _orf , _ = get_corr_filenames(split_d,args.output)
        _, _junc = get_class_junc_filenames(split_d,args.output)
        _info = get_pickle_filename(split_d,args.output)
        if args.include_ORF:
            with open(_orf) as h: f_faa.write(h.read())
        with open(_gtf) as h: f_gtf.write(h.read())
        with open(_fasta) as h: f_fasta.write(h.read())
        with open(f"{_junc}_tmp") as h:
            if i == 0:
                f_junc_temp.write(h.readline())
            else:
                h.readline()
            f_junc_temp.write(h.read())

        if args.isoform_hits:
            _iso_hits = f"{get_isoform_hits_name(split_d, args.output)}_tmp"
            with open(_iso_hits) as h:
                if i == 0:
                    isoform_hits.write(h.readline())
                else: # skip the header line
                    h.readline()
                isoform_hits.write(h.read())
        # Retrieving the isoform object and the junctions header
        with open(_info, 'rb') as h:
            isoforms_info.update(pickle.load(h))
            headers.append(pickle.load(h))
        if args.include_ORF:
            src_dir = os.path.join(split_d, "TD2")
            dst_dir = os.path.join(args.dir, "TD2", f"TD2_{i}")
            # If the directory is not removed, further runs will create nested directories if moving directly
            if os.path.exists(dst_dir):
                shutil.rmtree(dst_dir)

            # Now move the directory.
            shutil.move(src_dir, dst_dir)
    f_fasta.close()
    f_gtf.close()
    f_junc_temp.close()

    # Fix novel genes and classify FSM
    isoforms_info = rename_novel_genes(isoforms_info, args.novel_gene_prefix)
    isoforms_info = classify_fsm(isoforms_info)
    
    # Process expression data natively across all processed chunks
    isoforms_info = isoform_expression_info(isoforms_info, args.expression, args.short_reads, args.dir, corrFASTA, args.cpus)
    ## FL count file
    if args.fl_count:
        isoforms_info = full_length_quantification(args.fl_count, isoforms_info)
    isoforms_info,RTS_info = process_rts(isoforms_info,outputJuncPath,args.refFasta,genome_dict)

    fields_junc_cur = headers[0]
    if args.include_ORF:
        write_collapsed_GFF_with_CDS(isoforms_info, corrGTF, corrCDS_GTF_GFF)
    write_classification_output(isoforms_info, outputClassPath)
    write_junction_output(outputJuncPath, RTS_info, fields_junc_cur)
    #write omitted isoforms if requested minimum reference length is more than 0
    isoforms_info = write_omitted_isoforms(isoforms_info, args.dir, args.output,
                                            args.min_ref_len, args.is_fusion)
    #isoform hits to file if requested
    if args.isoform_hits:
        write_isoform_hits(args.dir, args.output, isoforms_info)
        isoform_hits.close()

    if args.include_ORF:
        f_faa.close()
    cleanup(outputClassPath, outputJuncPath)

    if args.report != 'skip':
        generate_report(args.saturation,args.report, outputClassPath, outputJuncPath)

# Run TUSCO benchmarking report if requested
    if hasattr(args, 'tusco') and args.tusco:
        # Pass both sample GTF (corrected) and reference GTF for IGV-like plots
        generate_tusco_report(args.tusco, outputClassPath, corrGTF, args.refGTF)
