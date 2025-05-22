import pandas as pd
import subprocess
import json
import re
from tqdm import tqdm

result_df = pd.read_csv("Tamer_Hosny_final.csv", encoding='utf-8-sig')

def classify_emotion(lyrics):
    if pd.isna(lyrics) or not lyrics.strip():
        return None

    prompt = f"""
You are a helpful assistant. Based on the Egyptian Arabic lyrics of a song, you must analyze the overall emotion of the singer. Your output must be a single JSON object with one key: "emotion".

Possible emotions include: Love, Sad, Heartbroken.

Here's an example:

Lyrics:
أنا ولا عارف أنا مالي ولا إيه اللي جرالي، بحبك
يا حبيبي بحبك وهتجنن عليك
أنا بقى حالي ما هو حالي ولا شاغل بالي يا عمري
غيرك يا عمري والله هموت عليك

حبيبي دي ليا أنا بتقولها وعارف معانيها
يبقى إنت عايزني أروح فيها يا حبيبي أنا
والبوسة دي مني أنا على إيدك
واحدة وعلى خدك مية واحدة
يا عم ومين قدك آه يا سيدي أنا

أنا ولا عارف أنا مالي ولا إيه اللي جرالي، بحبك
يا حبيبي بحبك وهتجنن عليك
أنا بقى حالي ما هو حالي ولا شاغل بالي يا عمري
غيرك يا عمري والله هموت عليك
أنا ولا عارف أنا مالي ولا إيه اللي جرالي، بحبك
يا حبيبي بحبك وهتجنن عليك
أنا بقى حالي ما هو حالي ولا شاغل بالي يا عمري
غيرك يا عمري والله هموت عليك

محتاج لكل اللي فيك خليني في حضنك قربني ليك
وريني شكل الغرام في عينيك، خليني معاك
خليتني أجي عندك وأهدي
يا جامد إنت، آه يا واد يا معدي
يا واخد إنت الحياة والروح وياك
محتاج لكل اللي فيك خليني في حضنك قربني ليك
وريني شكل الغرام في عينيك، خليني معاك
خليتني أجي عندك وأهدي
يا جامد إنت، آه يا واد يا معدي
يا واخد إنت الحياة والروح وياك

حبيبي دي ليا أنا بتقولها وعارف معانيها
يبقى إنت عايزني أروح فيها يا حبيبي أنا
والبوسة دي مني أنا على إيدك
واحدة وعلى خدك مية واحدة
يا عم ومين قدك آه يا سيدي أنا

أنا ولا عارف أنا مالي ولا إيه اللي جرالي، بحبك
يا حبيبي بحبك وهتجنن عليك
أنا بقى حالي ما هو حالي ولا شاغل بالي يا عمري
غيرك يا عمري والله هموت عليك
أنا ولا عارف أنا مالي ولا إيه اللي جرالي، بحبك
يا حبيبي بحبك وهتجنن عليك
أنا بقى حالي ما هو حالي ولا شاغل بالي يا عمري
غيرك يا عمري والله هموت عليك


Return:
{{"emotion": "Love"}}


Lyrics:
{lyrics}

Return the result in this exact format:
{{"emotion": "<One of the categories above>"}}
"""

    try:
        result = subprocess.run(
            ["ollama", "run", "qwen3:4b"],
            input=prompt.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30
        )

        output = result.stdout.decode().strip()
        match = re.search(r'\{.*?"emotion"\s*:\s*".*?"\s*\}', output)
        if match:
            emotion_dict = json.loads(match.group(0))
            res =emotion_dict.get("emotion", None)
            print(res)
            return res
        else:
            print(f"Failed to extract JSON from:\n{output}")
            return None

    except Exception as e:
        print(f"Error: {e}")
        return None


emotions = []
for lyrics in tqdm(result_df['Lyrics']):
    emotions.append(classify_emotion(lyrics))
result_df['emotion'] = emotions
result_df.to_csv("Tamer_Hosny_with_emotions.csv", index=False, encoding='utf-8-sig')
