import pandas as pd
from transformers import pipeline

# Local dataset path
df = pd.read_csv("../Dataset/sentiment_dataset.csv")

classifier = pipeline("sentiment-analysis")

results = classifier(df["Text"].tolist())

df["AI_Label"] = [r["label"] for r in results]
df["AI_Confidence"] = [round(r["score"], 3) for r in results]

df["AI_Label"] = df["AI_Label"].replace({
    "POSITIVE": "Positive",
    "NEGATIVE": "Negative"
})

df["Agreement"] = df["Sentiment"] == df["AI_Label"]

print("Agreement:", round(df["Agreement"].mean()*100,2), "%")

df.to_csv("../AI_Evaluation/ai_evaluation_results.csv", index=False)