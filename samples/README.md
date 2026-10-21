# Your labelled sample set

Put 10-20 representative handwritten files in `input/` and an exact typed transcription of each in
`ground_truth/` with the same base name:

    input/letter_01.pdf        ground_truth/letter_01.txt
    input/notes_page3.jpg      ground_truth/notes_page3.txt

Transcription rules (they must match what the model is asked to do):
- one line of handwriting = one line of text
- one blank line between paragraphs
- copy spelling and punctuation exactly; leave out crossed-out words
- for a multi-page PDF, transcribe all pages in order, separated by a blank line

Then run:

    python -m hwocr bench --inputs samples/input --gt samples/ground_truth --models qwen3-vl:8b,qwen3.5:9b
