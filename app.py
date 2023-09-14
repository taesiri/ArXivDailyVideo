import argparse
import os
import re
import string
from difflib import SequenceMatcher

import librosa
import moviepy.editor as mpy
import numpy as np
import requests
import torch
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline

# checkpoint = "openai/whisper-tiny"
# checkpoint = "openai/whisper-base"
checkpoint = "openai/whisper-small"

# We need to set alignment_heads on the model's generation_config (at least
# until the models have been updated on the hub).
# If you're going to use a different version of whisper, see the following
# for which values to use for alignment_heads:
# https://gist.github.com/hollance/42e32852f24243b748ae6bc1f985b13a

# whisper-tiny
# alignment_heads = [[2, 2], [3, 0], [3, 2], [3, 3], [3, 4], [3, 5]]
# whisper-base
# alignment_heads = [[3, 1], [4, 2], [4, 3], [4, 7], [5, 1], [5, 2], [5, 4], [5, 6]]
# whisper-small
alignment_heads = [
    [5, 3],
    [5, 9],
    [8, 0],
    [8, 4],
    [8, 7],
    [8, 8],
    [9, 0],
    [9, 7],
    [9, 9],
    [10, 5],
]

max_duration = 600  # seconds
fps = 60
video_width = 1920
video_height = 1080
margin_left = 1920 // 2 - 25
margin_right = 50
margin_top = 100
line_height = 65
total_lines = 14

background_image = Image.open("black_image.jpg")

font = ImageFont.truetype("Lato-Regular.ttf", 38)
title_font = ImageFont.truetype("Lato-Bold.ttf", 70)
id_font = ImageFont.truetype("Lato-Regular.ttf", 30)

text_color = (255, 200, 200)
highlight_color = (255, 255, 255)


LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
}

# language code lookup by name, with a few language aliases
TO_LANGUAGE_CODE = {
    **{language: code for code, language in LANGUAGES.items()},
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
}


if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    from transformers import (AutomaticSpeechRecognitionPipeline,
                              WhisperForConditionalGeneration,
                              WhisperProcessor)

    model = (
        WhisperForConditionalGeneration.from_pretrained(checkpoint).to("cuda").half()
    )
    processor = WhisperProcessor.from_pretrained(checkpoint)
    pipe = AutomaticSpeechRecognitionPipeline(
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        batch_size=8,
        torch_dtype=torch.float16,
        device="cuda:0",
    )
else:
    pipe = pipeline(model=checkpoint)

pipe.model.generation_config.alignment_heads = alignment_heads

chunks = []

start_chunk = 0
last_draws = None
last_image = None


def download_pdf(paper_id, save_dir="."):
    base_url = "https://arxiv.org/pdf/"
    pdf_url = os.path.join(base_url, f"{paper_id}.pdf")

    pdf_path = os.path.join(save_dir, f"{paper_id}.pdf")

    with requests.get(pdf_url, stream=True) as r:
        r.raise_for_status()
        with open(pdf_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    return pdf_path


def generate_modified_background(background_image_path, paper_id):
    # Load the original background
    background = Image.open(background_image_path)
    video_width = background.width
    video_height = background.height
    margin_left = 120
    id_bottom_margin = 30

    # Download the PDF of the paper
    pdf_path = download_pdf(paper_id)

    # Convert the first page of the PDF to an image
    pdf_images = convert_from_path(pdf_path)
    pdf_image = pdf_images[0]

    # Resize the PDF image to fit the height of the background
    pdf_image = pdf_image.resize(
        (int(pdf_image.width * video_height / pdf_image.height), video_height)
    )

    # Paste the PDF image onto the left side of the background
    background.paste(pdf_image, (0, 0))

    # Define fonts
    id_font = ImageFont.truetype("Lato-Regular.ttf", 30)

    draw = ImageDraw.Draw(background)

    # Calculate the width and height of the paper_id text
    id_width, id_height = draw.textsize(paper_id, font=id_font)

    # Define the position and dimensions for the rectangle background of paper_id
    rect_left = margin_left - 10
    rect_top = video_height - id_font.getsize(paper_id)[1] - id_bottom_margin
    rect_right = rect_left + id_width + 20
    rect_bottom = rect_top + id_height + 10

    # Draw a black rectangle for paper_id background
    draw.rectangle([rect_left, rect_top, rect_right, rect_bottom], fill=(0, 0, 0))

    # Draw the Arxiv ID on top of the black rectangle with white color
    draw.text(
        (margin_left, video_height - id_height - id_bottom_margin),
        paper_id,
        fill=(255, 255, 255),
        font=id_font,
    )

    # Save the modified background (optional)
    modified_background_path = "modified_background.jpg"
    background.save(modified_background_path)

    return modified_background_path


def make_frame(t, modified_background):
    global chunks, start_chunk, last_draws, last_image, total_lines

    image = modified_background.copy()  # Use the modified background
    draw = ImageDraw.Draw(image)

    space_length = draw.textlength(" ", font)
    x = margin_left
    y = margin_top + 20  # Add an additional margin from top to account for the title

    # Create a list of drawing commands
    draws = []
    for i in range(start_chunk, len(chunks)):
        chunk = chunks[i]
        chunk_start = chunk["timestamp"][0]
        chunk_end = chunk["timestamp"][1]
        if chunk_start > t:
            break
        if chunk_end is None:
            chunk_end = max_duration

        word = chunk["text"]
        word_length = draw.textlength(word + " ", font) - space_length

        if x + word_length >= video_width - margin_right:
            x = margin_left
            y += line_height

            # restart page when end is reached
            if y >= margin_top + line_height * total_lines:
                start_chunk = i
                break

        highlight = chunk_start <= t < chunk_end
        draws.append([x, y, word, word_length, highlight])

        x += word_length + space_length

    # If the drawing commands didn't change, then reuse the last image,
    # otherwise draw a new image
    if draws != last_draws:
        for x, y, word, word_length, highlight in draws:
            if highlight:
                # Set text color to black for highlighted word
                color = (0, 0, 0)
                draw.rectangle(
                    [x, y, x + word_length, y + line_height],
                    fill=(255, 255, 0),  # yellow background for highlighted word
                )
            else:
                color = text_color

            draw.text((x, y), word, fill=color, font=font)

        last_image = np.array(image)
        last_draws = draws

    return last_image


def preprocess_text(text):
    """Preprocess the text by making it lowercase and removing punctuation."""
    return "".join(ch for ch in text.lower() if ch not in string.punctuation).split()


def preprocess_word(word):
    """Preprocess a single word by removing punctuation and converting to lowercase."""
    return re.sub(r"[^a-zA-Z0-9]", "", word).lower()


def lcs(X, Y):
    """Find the longest common subsequence of X and Y"""
    m = len(X)
    n = len(Y)
    L = [[0] * (n + 1) for i in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    # Following code is used to print LCS
    index = L[m][n]
    lcs = [""] * (index + 1)
    lcs[index] = ""
    i = m
    j = n
    while i > 0 and j > 0:
        if X[i - 1] == Y[j - 1]:
            lcs[index - 1] = (X[i - 1], i - 1, j - 1)
            i -= 1
            j -= 1
            index -= 1
        elif L[i - 1][j] > L[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return lcs[:-1]


def robust_match_whisper_with_gt_v9(whisper_output, gt):
    """Robustly match words from the Whisper output with words in the ground truth using the LCS algorithm."""

    # Convert whisper output to a list of words
    whisper_words = [word_data["text"].strip() for word_data in whisper_output]
    whisper_processed = [preprocess_word(word) for word in whisper_words]

    # Split the ground truth into words
    gt_words = gt.split()
    gt_processed = [preprocess_word(word) for word in gt_words]

    common_sequence = lcs(gt_processed, whisper_processed)
    matched_words = []
    last_known_timestamp = None

    gt_pointer = 0
    for word, gt_index, whisper_index in common_sequence:
        # First, add any missing GT words before the current matched word
        while gt_pointer < gt_index:
            matched_words.append(
                {
                    "text": gt_words[gt_pointer],
                    "timestamp": last_known_timestamp
                    if last_known_timestamp
                    else whisper_output[whisper_index]["timestamp"],
                }
            )
            gt_pointer += 1

        # Now, add the matched word with its timestamp
        matched_words.append(
            {
                "text": gt_words[gt_pointer],
                "timestamp": whisper_output[whisper_index]["timestamp"],
            }
        )
        last_known_timestamp = whisper_output[whisper_index]["timestamp"]
        gt_pointer += 1

    # If there are any remaining words in the GT, append them with the last known timestamp
    while gt_pointer < len(gt_words):
        matched_words.append(
            {
                "text": gt_words[gt_pointer],
                "timestamp": last_known_timestamp
                if last_known_timestamp
                else whisper_output[0]["timestamp"],
            }
        )
        gt_pointer += 1

    return matched_words


def redistribute_timestamps(matched_output):
    """
    Redistribute timestamps for consecutive words that have the same timestamp.
    """
    modified_output = []
    n = len(matched_output)

    # A function to uniformly distribute timestamps
    def distribute_time(start_time, end_time, count):
        interval = (end_time - start_time) / (count + 1)
        return [
            (start_time + interval * i, start_time + interval * (i + 1))
            for i in range(count)
        ]

    i = 0
    while i < n:
        # If current word and next word have the same timestamp
        if (
            i < n - 1
            and matched_output[i]["timestamp"] == matched_output[i + 1]["timestamp"]
        ):
            count = 1
            # Count how many consecutive words have the same timestamp
            while (
                i + count < n - 1
                and matched_output[i]["timestamp"]
                == matched_output[i + count + 1]["timestamp"]
            ):
                count += 1

            # Identify the time before and after the block of words with the same timestamp
            start_time = matched_output[i - 1]["timestamp"][1] if i > 0 else 0
            end_time = (
                matched_output[i + count + 1]["timestamp"][0]
                if i + count + 1 < n
                else matched_output[i + count]["timestamp"][1]
            )

            # Distribute the time uniformly
            new_timestamps = distribute_time(start_time, end_time, count + 1)

            for j in range(count + 1):
                modified_output.append(
                    {
                        "text": matched_output[i + j]["text"],
                        "timestamp": new_timestamps[j],
                    }
                )

            i += count + 1
        else:
            modified_output.append(matched_output[i])
            i += 1

    return modified_output


def predict(paper_id, language=None):
    global chunks, start_chunk, last_draws, last_image, background_image

    # Fetch the title from the Arxiv API
    title = get_arxiv_title(paper_id)
    if not title:
        return "Error: Could not fetch the paper title."

    background_image_path = "black_image.jpg"

    # Create a modified background with the title and ID
    modified_background_path = generate_modified_background(
        background_image_path, paper_id
    )
    modified_background_image = Image.open(modified_background_path)

    # Download the abstract and audio
    abstract_path, audio_path = download_data(paper_id)

    # Append the title to the abstract
    with open(abstract_path, "r") as f:
        abstract = f.read()

    title = title.replace("\n", " ")
    gt_text = title + ".\n\n" + abstract
    with open(abstract_path, "w") as f:
        f.write(gt_text)

    start_chunk = 0
    last_draws = None
    last_image = None

    audio_data, sr = librosa.load(audio_path, mono=True)
    duration = librosa.get_duration(y=audio_data, sr=sr)
    duration = min(max_duration, duration)
    audio_data = audio_data[: int(duration * sr)]

    if language is not None:
        pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(
            language=language, task="transcribe"
        )

    # Run Whisper to get word-level timestamps
    audio_inputs = librosa.resample(
        audio_data, orig_sr=sr, target_sr=pipe.feature_extractor.sampling_rate
    )
    output = pipe(
        audio_inputs,
        chunk_length_s=30,
        stride_length_s=[4, 2],
        return_timestamps="word",
    )
    chunks = output["chunks"]

    # Match Whisper output with ground truth
    chunks = robust_match_whisper_with_gt_v9(chunks, gt_text)
    chunks = redistribute_timestamps(chunks)

    # Create the video
    clip = mpy.VideoClip(
        lambda x: make_frame(x, modified_background_image), duration=duration
    )  # Modified this line to pass the modified background
    audio_clip = mpy.AudioFileClip(audio_path).set_duration(duration)
    clip = clip.set_audio(audio_clip)
    video_path = f"{paper_id}_video.mp4"
    clip.write_videofile(video_path, fps=fps, codec="libx264", audio_codec="aac")

    # Clean up the downloaded abstract and audio files
    os.remove(abstract_path)
    os.remove(audio_path)

    return video_path


def get_arxiv_title(paper_id):
    base_url = "http://export.arxiv.org/api/query?id_list="
    response = requests.get(base_url + paper_id)
    if response.status_code == 200:
        start = response.text.find("<title>") + 7
        end = response.text.find("</title>", start)
        title = response.text[start:end]
        return title
    else:
        print(f"Failed to get title for paper ID: {paper_id}")
        return None


def download_data(paper_id, save_dir="."):
    base_url = "https://huggingface.co/datasets/taesiri/arxiv_audio/"
    abstract_url = os.path.join(base_url, "raw/main/abstract", f"{paper_id}.txt")
    audio_url = os.path.join(base_url, "resolve/main/audio", f"{paper_id}.mp3")

    abstract_path = os.path.join(save_dir, f"{paper_id}_abstract.txt")
    audio_path = os.path.join(save_dir, f"{paper_id}.mp3")

    with requests.get(abstract_url, stream=True) as r:
        r.raise_for_status()
        with open(abstract_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    with requests.get(audio_url, stream=True) as r:
        r.raise_for_status()
        with open(audio_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    return abstract_path, audio_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate a video with subtitled audio using Whisper."
    )
    parser.add_argument("--pid", type=str, help="Arxiv paper ID.")
    parser.add_argument(
        "--language",
        type=str,
        default=None,  # set default to None
        choices=sorted(list(TO_LANGUAGE_CODE.keys())),
        help="Language of the audio content. Optional.",
    )
    args = parser.parse_args()

    output_path = predict(args.pid, args.language)
    print(f"Generated video saved at: {output_path}")


if __name__ == "__main__":
    main()
