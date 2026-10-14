# data/papers/

Place your research-paper PDF files here before running smoke tests with `--real-papers`.

## Recommended open-access arXiv papers

The three papers used in the synthetic smoke tests are all freely available on arXiv.
Download them with the commands below (requires `curl` or your browser):

### 1. Attention Is All You Need (Vaswani et al., 2017)
```bash
curl -L "https://arxiv.org/pdf/1706.03762" -o "Attention_Is_All_You_Need.pdf"
```

### 2. BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)
```bash
curl -L "https://arxiv.org/pdf/1810.04805" -o "BERT.pdf"
```

### 3. Deep Residual Learning for Image Recognition (He et al., 2015)
```bash
curl -L "https://arxiv.org/pdf/1512.03385" -o "Deep_Residual_Learning.pdf"
```

## One-liner (downloads all three)
```bash
curl -L "https://arxiv.org/pdf/1706.03762" -o data/papers/Attention_Is_All_You_Need.pdf \
  && curl -L "https://arxiv.org/pdf/1810.04805" -o data/papers/BERT.pdf \
  && curl -L "https://arxiv.org/pdf/1512.03385" -o data/papers/Deep_Residual_Learning.pdf
```

## Running smoke tests against real PDFs
```bash
python tests/smoke_test_coordinator.py --real-papers
```

## Notes
- Any PDF research paper can be placed here; the tool is not limited to these three.
- Files must have the `.pdf` extension.
- The system extracts title and year from the PDF metadata or first page automatically.
- For best extraction quality, use text-based PDFs (not scanned images).
