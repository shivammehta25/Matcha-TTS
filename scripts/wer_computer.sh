#!/bin/bash
# Run from root folder with: bash scripts/wer_computer.sh


root_folder=${1:-"dur_wer_computation"}
echo "Running WER computation for Duration predictors"
cmd="wercompute -r ${root_folder}/reference_transcripts/ -i ${root_folder}/lj_fm_output_transcriptions/"
# echo $cmd
echo "LJ"
echo "==================================="
echo "Flow Matching"
$cmd
echo "-----------------------------------"

echo "LJ Determinstic"
cmd="wercompute -r ${root_folder}/reference_transcripts/ -i ${root_folder}/lj_det_output_transcriptions/"
$cmd
echo "-----------------------------------"

echo "Cormac"
echo "==================================="
echo "Cormac Flow Matching"
cmd="wercompute -r ${root_folder}/reference_transcripts/ -i ${root_folder}/fm_output_transcriptions/"
$cmd
echo "-----------------------------------"

echo "Cormac Determinstic"
cmd="wercompute -r ${root_folder}/reference_transcripts/ -i ${root_folder}/det_output_transcriptions/"
$cmd
echo "-----------------------------------"