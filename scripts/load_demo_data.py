"""
Demo data loader — populates the paper store with sample papers
so the app can be tested immediately without API calls.

Run:  python -m scripts.load_demo_data
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.paper_store import paper_store

DEMO_PAPERS = [
    {
        "paper_id": "demo-attention-2017",
        "title": "Attention Is All You Need",
        "abstract": (
            "The dominant sequence transduction models are based on complex recurrent or "
            "convolutional neural networks that include an encoder and a decoder. The best "
            "performing models also connect the encoder and decoder through an attention mechanism. "
            "We propose a new simple network architecture, the Transformer, based solely on attention "
            "mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two "
            "machine translation tasks show these models to be superior in quality while being more "
            "parallelizable and requiring significantly less time to train."
        ),
        "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit"],
        "year": 2017,
        "citations": 95000,
        "url": "https://arxiv.org/abs/1706.03762",
        "pdf_url": "https://arxiv.org/pdf/1706.03762",
        "topics": ["Natural Language Processing", "Machine Learning", "Deep Learning"],
        "source": "demo",
        "indexed": False,
    },
    {
        "paper_id": "demo-bert-2018",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "abstract": (
            "We introduce a new language representation model called BERT, which stands for "
            "Bidirectional Encoder Representations from Transformers. Unlike recent language "
            "representation models, BERT is designed to pre-train deep bidirectional representations "
            "from unlabeled text by jointly conditioning on both left and right context in all layers. "
            "As a result, the pre-trained BERT model can be fine-tuned with just one additional output "
            "layer to create state-of-the-art models for a wide range of tasks."
        ),
        "authors": ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"],
        "year": 2018,
        "citations": 78000,
        "url": "https://arxiv.org/abs/1810.04805",
        "pdf_url": "https://arxiv.org/pdf/1810.04805",
        "topics": ["Natural Language Processing", "Pre-training", "Transformers"],
        "source": "demo",
        "indexed": False,
    },
    {
        "paper_id": "demo-gpt3-2020",
        "title": "Language Models are Few-Shot Learners",
        "abstract": (
            "Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by "
            "pre-training on a large corpus of text followed by fine-tuning on a specific task. "
            "We show that scaling up language models greatly improves task-agnostic, few-shot performance, "
            "sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches. "
            "We train GPT-3, an autoregressive language model with 175 billion parameters, and test its "
            "performance in the few-shot setting."
        ),
        "authors": ["Tom Brown", "Benjamin Mann", "Nick Ryder", "Melanie Subbiah", "Jared Kaplan"],
        "year": 2020,
        "citations": 35000,
        "url": "https://arxiv.org/abs/2005.14165",
        "pdf_url": "https://arxiv.org/pdf/2005.14165",
        "topics": ["Natural Language Processing", "Large Language Models", "Few-Shot Learning"],
        "source": "demo",
        "indexed": False,
    },
    {
        "paper_id": "demo-rag-2020",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "abstract": (
            "Large pre-trained language models have been shown to store factual knowledge in their "
            "parameters, and achieve state-of-the-art results when fine-tuned on downstream NLP tasks. "
            "We explore a general-purpose fine-tuning recipe for retrieval-augmented generation (RAG) "
            "models — models that combine pre-trained parametric and non-parametric memory for language "
            "generation. We introduce RAG models where the parametric memory is a pre-trained seq2seq "
            "model and the non-parametric memory is a dense vector index of Wikipedia, accessed with "
            "a pre-trained neural retriever."
        ),
        "authors": ["Patrick Lewis", "Ethan Perez", "Aleksandra Piktus", "Fabio Petroni"],
        "year": 2020,
        "citations": 9800,
        "url": "https://arxiv.org/abs/2005.11401",
        "pdf_url": "https://arxiv.org/pdf/2005.11401",
        "topics": ["Retrieval-Augmented Generation", "Knowledge-Intensive NLP", "Open-Domain QA"],
        "source": "demo",
        "indexed": False,
    },
    {
        "paper_id": "demo-diffusion-2020",
        "title": "Denoising Diffusion Probabilistic Models",
        "abstract": (
            "We present high quality image synthesis results using diffusion probabilistic models, "
            "a class of latent variable models inspired by considerations from nonequilibrium thermodynamics. "
            "Our best results are obtained by training on a weighted variational bound designed according to "
            "a novel connection between diffusion probabilistic models and denoising score matching with "
            "Langevin dynamics, and our models naturally admit a progressive lossy decompression scheme."
        ),
        "authors": ["Jonathan Ho", "Ajay Jain", "Pieter Abbeel"],
        "year": 2020,
        "citations": 12000,
        "url": "https://arxiv.org/abs/2006.11239",
        "pdf_url": "https://arxiv.org/pdf/2006.11239",
        "topics": ["Generative Models", "Image Synthesis", "Deep Learning"],
        "source": "demo",
        "indexed": False,
    },
    {
        "paper_id": "demo-llama2-2023",
        "title": "Llama 2: Open Foundation and Fine-Tuned Chat Models",
        "abstract": (
            "We develop and release Llama 2, a collection of pretrained and fine-tuned large language "
            "models (LLMs) ranging in scale from 7 billion to 70 billion parameters. Our fine-tuned LLMs, "
            "called Llama 2-Chat, are optimized for dialogue use cases. Our models outperform open-source "
            "chat models on most benchmarks we tested, and based on our human evaluations for helpfulness "
            "and safety, may be a suitable substitute for closed-source models."
        ),
        "authors": ["Hugo Touvron", "Louis Martin", "Kevin Stone"],
        "year": 2023,
        "citations": 8200,
        "url": "https://arxiv.org/abs/2307.09288",
        "pdf_url": "https://arxiv.org/pdf/2307.09288",
        "topics": ["Large Language Models", "Open Source", "RLHF"],
        "source": "demo",
        "indexed": False,
    },
]


def load_demo_data():
    loaded = 0
    for paper in DEMO_PAPERS:
        try:
            paper_store.add(paper)
            loaded += 1
            print(f"  OK  {paper['title'][:60]}")
        except Exception as e:
            print(f"  FAIL: {e}")
    print(f"\nLoaded {loaded}/{len(DEMO_PAPERS)} demo papers.")


if __name__ == "__main__":
    print("Loading demo papers into store…")
    load_demo_data()
    print("\nDone! Start the backend and visit the dashboard.")
