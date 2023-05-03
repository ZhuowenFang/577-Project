import openai
import pandas as pd
import tqdm
openai.api_key = "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"

def chat(inp, role="user"):
    inp = str(inp) + "Rate the politeness of the previous sentence on a scale of 1 to 10, where 1 represents extremely impolite and 10 represents extremely polite, just give me the number, no need to explain."
    message_history = [{"role": role, "content": f"{inp}"}]
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message_history
    )
    reply_content = completion.choices[0].message.content
    return reply_content
df = pd.read_csv("Tweets.csv")
temp = 1
for i in tqdm.tqdm(df["text"]):
    if temp >= 11318:
        result = pd.DataFrame({"text": [i], "label": [chat(i)]})
        result.to_csv("output.csv", index=False, header=False, mode="a")
    temp += 1