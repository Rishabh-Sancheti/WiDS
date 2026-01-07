from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

text = (
    "Artificial intelligence has seen rapid growth over the past decade, "
    "impacting industries such as healthcare, finance, education, and transportation. "
    "Machine learning models are now capable of performing tasks that once required human expertise, "
    "including image recognition, natural language understanding, and decision-making. "
    "Despite these advances, concerns remain about ethical use, bias, and job displacement. "
    "As AI continues to evolve, responsible development and regulation will play a crucial role "
    "in ensuring its benefits are shared widely across society."
)

summary = summarizer(
    text,
    min_length=40,
    max_length=80,
    do_sample=False
)[0]["summary_text"]

original_length = len(text.split())
summary_length = len(summary.split())

print("Original text length (words):", original_length)
print("\nSummary:")
print(summary)
print("\nSummary length (words):", summary_length)
