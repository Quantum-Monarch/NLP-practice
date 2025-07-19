

def parse_daily_dialog(filepath):
    pairs = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:  # skip empty lines
                continue
            utterances = [u.strip() for u in line.split("__eou__") if u.strip()]
            context = ""
            for utt in range(len(utterances)-1):
                if utt % 2 == 0:
                    prompt="user:"+utterances[utt]
                    response="bot:"+utterances[utt+1]
                else:
                    prompt="bot:"+utterances[utt]
                    response="user:"+utterances[utt+1]
                if context == "":
                    pairs.append(("", prompt, response))
                else:
                    pairs.append((context, prompt, response))
                if utt > 2:
                    context=context.split("|",1)[1]
                print(type(context),context)
                print(type(prompt),prompt)
                context = (context + "|" + prompt).strip()
    return pairs
data=parse_daily_dialog("dialogues_train.txt")
data2=data[:5000]
print(data[:10])