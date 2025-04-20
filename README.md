# SimCLR Self-Supervised Learning Example

An example of running SimCLR on your unlabeled images. 
Based on PyTorch code. For supervised learning, the classifier sections after SimCLR training.

## How to use

1. **Clone the repository**:git clone https://github.com/your-username/simclr-example.git
cd simclr-example
2. **Place the images without markings in the folder** `data/unlabelled/class_x/` (change the structure to your own, but it is advisable to leave it ImageFolder-compatible).
3. **Install dependencies**:
4. **Start training**: python src/main.py
5. After pretraining for further training - add a classifier, freeze the backbone.

---

## Fine-tuning
(An example of additional training is at the end of main.py.)
