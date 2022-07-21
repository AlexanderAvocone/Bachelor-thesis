# How to batching with HTCondor

1.) Scripts in arguments.txt

2.) bash script (job.sh) to call the scripts

3.) define runtime, RAM, etc.

- to submit: condor_submit submit.jdl
- to monitor: condor_q
- to kill: condor_rm 'id'
