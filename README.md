# **Fincheck – Confidence-Aware Cheque Digit Validation System**

**Next.js (Frontend) · FastAPI + PyTorch (Backend)**

---

## 1. Overview

**Fincheck** is a **full-stack fintech verification system** designed to detect **incorrect, ambiguous, or risky handwritten digits** in financial documents such as **bank cheques**.

Unlike conventional OCR systems that *always output a digit*, **Fincheck is explicitly risk-aware**.
Its primary goal is **to identify when a digit should *not* be trusted**.

> **Core Principle**
> *In financial systems, a wrong prediction is more dangerous than no prediction.*

---

## 2. Problem Statement

Traditional OCR pipelines optimize for **maximum accuracy**, often producing confident outputs even when the input digit is ambiguous, noisy, or out-of-distribution.

In financial workflows (cheques, account numbers, amounts):

* Silent misclassification can lead to **monetary loss**
* Ambiguous digits (e.g., 3 vs 5, 1 vs 7) are **high-risk**
* Systems rarely expose *uncertainty*

**Fincheck addresses this gap** by explicitly modeling **confidence and ambiguity**, rather than forcing a single prediction.

---

## 3. Key Features

### 3.1 Image-Only Cheque Digit Verification

* Accepts **single or multiple handwritten digits**
* Robust to **noise, scans, camera images, skew, and blur**
* Supports **PNG / JPG / JPEG** formats
* No dependency on typed text input

---

### 3.2 Confidence-Aware Digit Validation

Each detected digit is classified into one of three states:

* **VALID** – High confidence, safe to accept
* **AMBIGUOUS** – Multiple plausible digits
* **INVALID** – Low confidence / unreliable / out-of-distribution

This prevents unsafe automation in financial decision pipelines.

---

### 3.3 Position-Level Error Reporting

For every digit position, the system reports:

* Digit index / position
* Predicted value
* Confidence score (%)
* Alternative plausible digits (if ambiguous)

This allows **human-in-the-loop verification**.

---

### 3.4 MNIST-Based Shape Verification (No Blind OCR)

* Uses **pretrained MNIST CNN models (.pth)**
* Converts cheque digits into **MNIST-style normalized 28×28 images**
* Rejects inputs that fall **outside the learned digit manifold**
* Avoids forced predictions common in OCR engines

---

## 4. Why MNIST Is Used (Important Clarification)

MNIST is **not** used to “recognize cheques”.

It is used as a **digit shape validity reference** to evaluate:

* Whether a digit resembles known handwritten digit distributions
* Whether the digit is ambiguous or unsafe
* Whether the model should abstain from prediction

This design explicitly **prioritizes risk minimization over raw accuracy**.

---

## 5. System Architecture

```text
User (Browser)
   ↓
Next.js Frontend (Bun)
   ↓ API Requests
FastAPI Backend
   ↓
Image Preprocessing (OpenCV)
   ↓
Digit Segmentation (Connected Components)
   ↓
MNIST Normalization (28×28 + Center of Mass)
   ↓
Confidence-Aware MNIST Inference (PyTorch)
   ↓
VALID / AMBIGUOUS / INVALID Decision
```

---

## 6. Tech Stack

### 6.1 Frontend

* Next.js (App Router)
* TypeScript
* Tailwind CSS
* Bun
* Canvas-based image rendering
* Fetch API

---

### 6.2 Backend

* FastAPI
* PyTorch
* OpenCV
* NumPy / SciPy
* PIL
* Torchvision
* Tesseract
  *(Used only for `/verify`, not for image-only digit validation)*

---

## 7. Prerequisites

### Common

* Git
* Stable internet connection

### Frontend

* **Bun (mandatory)**

### Backend

* **Python 3.10 – 3.12**
* pip
* Optional: CUDA for GPU inference

---

## 8. Installing Bun (Frontend)

### macOS / Linux

```bash
curl -fsSL https://bun.sh/install | bash
source ~/.zshrc   # or ~/.bashrc
bun --version
```

### Windows (PowerShell)

```powershell
powershell -c "irm bun.sh/install.ps1 | iex"
```

Restart terminal and verify:

```powershell
bun --version
```

---

## 9. Clone Repository

```bash
git clone <YOUR_REPO_URL>
cd fincheck-next
```

---

## 10. Backend Setup (FastAPI + PyTorch)

### Navigate to backend

```bash
cd fintech-backend
```

### Create virtual environment

```bash
python -m venv venv
```

Activate it:

```bash
# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

---

### Install dependencies

```bash
pip install -r requirements.txt
pip install scipy
```

---

### Download pretrained models

```bash
python download_models.py
```

---

### Start backend server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Backend runs at:

```
http://127.0.0.1:8000
```

> **Note (Windows):**
> The Microsoft C++ Build Tools must be installed for `pytesseract` to function correctly.

---

## 11. Backend API Endpoints

| Endpoint             | Method | Description                        |
| -------------------- | ------ | ---------------------------------- |
| `/verify-digit-only` | POST   | Image-only cheque digit validation |
| `/verify`            | POST   | OCR + typed text verification      |
| `/run`               | POST   | Single MNIST image inference       |
| `/run-dataset`       | POST   | Dataset-level MNIST evaluation     |

---

## 12. Frontend Setup (Next.js + Bun)

### Navigate to frontend

```bash
cd ../fintech-frontend
```

### Install dependencies

```bash
bun install
```

---

## 13. Environment Variables (Frontend)

Create `.env` file:

```bash
touch .env
```

### `.env` (DO NOT COMMIT)

```env
# Database
MONGODB_URI=mongodb+srv://<username>:<password>@<cluster>/<db>

# Authentication
BETTER_AUTH_SECRET=your_secret_here
BETTER_AUTH_URL=http://localhost:3000

# Backend
INFERENCE_API_URL=http://127.0.0.1:8000

# OAuth (optional)
GITHUB_CLIENT_ID=your_client_id
GITHUB_CLIENT_SECRET=your_client_secret
```

Add to `.gitignore`:

```gitignore
.env
.env.local
```

---

## 14. Run Frontend (Development)

```bash
bun run dev
```

Frontend available at:

```
http://localhost:3000
```

---

## 15. Example Output (Cheque Digit Validation)

```text
Verdict: AMBIGUOUS
Detected Digits: 709

Position 1
Status: VALID
Predicted: 7
Confidence: 97%

Position 2
Status: VALID
Predicted: 0
Confidence: 97.65%

Position 3
Status: AMBIGUOUS
Predicted: 9
Confidence: 72.5%
Possible values: 9, 3, 5
```

---

## 16. Output Screenshots

This section documents **real system outputs** for transparency and evaluation.

### Included Screenshots

* Uploaded cheque digit image
* Segmented digit bounding boxes
* MNIST-normalized 28×28 digit samples
* Frontend validation result view
* Confidence and ambiguity indicators

> Screenshots should be placed under:

```text
/docs/screenshots/
```

Recommended filenames:

```text
input_cheque.png
digit_segmentation.png
mnist_normalization.png
validation_result.png
```

---

## 17. Evaluation Philosophy

Fincheck **does not aim to maximize accuracy**.

Instead, it minimizes **financial risk** by:

* Abstaining on low-confidence predictions
* Flagging ambiguous digits
* Exposing uncertainty explicitly
* Supporting human verification

This makes it suitable for **real-world financial workflows**, not just benchmarks.

---

## 18. License

This project is intended for **academic, research, and demonstration purposes** only.
