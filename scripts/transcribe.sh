echo "Transcribing"

whispertranscriber -i lj_det_output -o lj_det_output_transcriptions -f

whispertranscriber -i lj_fm_output -o lj_fm_output_transcriptions -f
wercompute -r dur_wer_computation/reference_transcripts/ -i lj_det_output_transcriptions
wercompute -r dur_wer_computation/reference_transcripts/ -i lj_fm_output_transcriptions