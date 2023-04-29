import logging
from faster_whisper import WhisperModel
import datetime

model_size = "small.en"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="int8")

def timestamp_to_formatted(timestamp):
    # Convert timestamp in seconds to timedelta object
    td = datetime.timedelta(seconds=timestamp)
    # Convert timedelta object to datetime object with a zero date
    dt = datetime.datetime(1, 1, 1) + td
    # Format datetime object as HH:mm:ss.SSS string
    return dt.strftime('%H:%M:%S.%f')[:-4]
    
segments, _ = model.transcribe("test.wav", word_timestamps=True, no_speech_threshold=0.3)
segments = list(segments)  # The transcription will actually run here.

print("[Script Info]")
print("Title: ")
print("ScriptType: v4.00+")
print("PlayDepth: 0")
print("ScaledBorderAndShadow: Yes")

print("[V4+ Styles]")
print("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding")
print("Style: Default,Arial,20,&H00FFFFFF,&H0000FFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1,1,2,10,10,10,1")


for segment in segments:
    for word in segment.words:
        #print(word.start)
        fm_start = timestamp_to_formatted(word.start)
        fm_end = timestamp_to_formatted(word.end)
        print("Dialogue: 0,"+fm_start+","+fm_end+",*Default,NTP,0,0,0,,"+word.word)
        

logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)
