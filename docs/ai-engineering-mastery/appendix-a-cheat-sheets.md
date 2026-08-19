# Appendix A — Cheat Sheets

## Python Quick Reference

```python
# ─── Data Types ────────────────────────────
x = 42              # int
y = 3.14            # float
s = "hello"         # str
lst = [1, 2, 3]     # list (mutable)
tup = (1, 2)        # tuple (immutable)
d = {"a": 1}        # dict
st = {1, 2, 3}      # set
b = True            # bool

# ─── List Comprehensions ───────────────────
squares = [x**2 for x in range(10)]
evens = [x for x in range(20) if x % 2 == 0]
flat = [n for row in matrix for n in row]

# ─── String Methods ────────────────────────
"Hello".lower()          # "hello"
"hello".upper()          # "HELLO"
"  hi  ".strip()         # "hi"
"a,b,c".split(",")       # ["a","b","c"]
",".join(["a","b"])      # "a,b"
"hello".replace("l","")  # "heo"

# ─── Dictionary Methods ────────────────────
d.get("key", default)
d.keys(), d.values(), d.items()
d.update(other_dict)
{k: v for k, v in d.items() if v > 0}

# ─── f-strings ─────────────────────────────
name = "Alice"
f"Hello {name}, you are {age} years old"
f"{pi:.2f}"           # 3.14
f"{1000000:,}"       # 1,000,000
f"{0.15:.1%}"        # 15.0%

# ─── File I/O ──────────────────────────────
with open("file.txt") as f:
    text = f.read()
    # or: lines = f.readlines()
    # or: for line in f: ...

with open("out.txt", "w") as f:
    f.write("content")

# ─── Exception Handling ───────────────────
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"Error: {e}")
finally:
    print("Always runs")

# ─── Classes ───────────────────────────────
class Dog:
    def __init__(self, name):
        self.name = name
    def bark(self):
        return f"{self.name} says woof!"

d = Dog("Rex")
d.bark()  # "Rex says woof!"
```

## NumPy Cheat Sheet

```python
import numpy as np

# ─── Creating Arrays ──────────────────────
np.array([1, 2, 3])           # 1D array
np.zeros((3, 4))              # 3x4 zeros
np.ones((2, 2))                # 2x2 ones
np.arange(0, 10, 2)            # [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)           # [0, 0.25, 0.5, 0.75, 1]
np.random.rand(3, 3)           # random 3x3
np.random.randn(3, 3)          # standard normal
np.eye(3)                      # identity matrix

# ─── Array Properties ─────────────────────
a.shape     # (rows, cols)
a.ndim      # number of dimensions
a.size      # total elements
a.dtype     # data type
a.T         # transpose

# ─── Indexing & Slicing ───────────────────
a[0]            # first row
a[-1]           # last row
a[1:3]          # rows 1-2
a[:, 0]         # first column
a[a > 5]        # boolean mask
a.reshape(3, 4) # reshape

# ─── Math Operations ──────────────────────
a + b           # element-wise add
a * b           # element-wise multiply
a @ b           # matrix multiply
np.dot(a, b)    # dot product
np.sum(a)       # sum all
np.sum(a, axis=0)  # sum columns
np.mean(a)      # mean
np.std(a)       # standard deviation
np.max(a, axis=1)  # max per row
np.exp(a)       # e^a
np.log(a)       # ln(a)
np.sqrt(a)      # square root

# ─── Broadcasting ─────────────────────────
a + 5           # add scalar to every element
a + [1, 2, 3]   # add vector to every row
```

## Pandas Cheat Sheet

```python
import pandas as pd

# ─── Creating DataFrames ──────────────────
df = pd.DataFrame({"name": ["A","B"], "age": [25, 30]})
df = pd.read_csv("data.csv")
df = pd.read_json("data.json")

# ─── Inspection ───────────────────────────
df.head()        # first 5 rows
df.tail()        # last 5 rows
df.shape         # (rows, cols)
df.columns       # column names
df.dtypes        # data types
df.describe()    # statistics
df.info()        # summary

# ─── Selecting ────────────────────────────
df["col"]        # one column (Series)
df[["c1", "c2"]] # multiple columns
df.iloc[0]       # first row by position
df.loc[0]        # first row by label
df.iloc[0:5, 0:2]  # rows 0-4, cols 0-1

# ─── Filtering ────────────────────────────
df[df["age"] > 25]
df[(df["age"] > 25) & (df["city"] == "NYC")]
df.query("age > 25 and city == 'NYC'")

# ─── Grouping ─────────────────────────────
df.groupby("city")["salary"].mean()
df.groupby(["city", "dept"])["salary"].agg(["mean", "max"])
df.groupby("city").size()

# ─── Missing Data ────────────────────────
df.isnull().sum()      # count nulls
df.dropna()             # drop rows with nulls
df.fillna(0)            # fill nulls with 0
df["col"].fillna(df["col"].mean())  # fill with mean

# ─── Apply ────────────────────────────────
df["col"].apply(lambda x: x * 2)
df["new"] = df["col"].map({"A": 1, "B": 2})

# ─── Merge & Join ─────────────────────────
pd.merge(df1, df2, on="id")
pd.merge(df1, df2, on="id", how="left")
pd.concat([df1, df2])

# ─── Output ───────────────────────────────
df.to_csv("out.csv", index=False)
df.to_json("out.json")
```

## Scikit-learn Cheat Sheet

```python
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix

# ─── Split ───────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ─── Scale ───────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ─── Train ───────────────────────────────
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ─── Predict ─────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# ─── Evaluate ────────────────────────────
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
confusion_matrix(y_test, y_pred)

# ─── Cross-Validation ────────────────────
scores = cross_val_score(model, X, y, cv=5)
print(f"CV: {scores.mean():.3f} ± {scores.std():.3f}")

# ─── Grid Search ─────────────────────────
params = {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 10]}
grid = GridSearchCV(model, params, cv=5, scoring="accuracy")
grid.fit(X_train, y_train)
print(grid.best_params_)
```

## PyTorch Cheat Sheet

```python
import torch
import torch.nn as nn
import torch.optim as optim

# ─── Tensors ─────────────────────────────
t = torch.tensor([[1, 2], [3, 4]])
torch.zeros(3, 4)
torch.ones(2, 2)
torch.randn(3, 3)           # standard normal
t.shape, t.dtype, t.device
t.to("cuda")               # move to GPU

# ─── Operations ──────────────────────────
a + b, a * b               # element-wise
torch.matmul(a, b)         # matrix multiply
a.sum(), a.mean(), a.max()
a.reshape(2, -1)
a.unsqueeze(0)             # add dimension
a.squeeze()                # remove size-1 dims

# ─── Model Definition ────────────────────
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 10),
    nn.Softmax(dim=1)
)

# ─── Training Loop ───────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()         # clear gradients
        outputs = model(X_batch)       # forward pass
        loss = criterion(outputs, y_batch)  # compute loss
        loss.backward()               # backprop
        optimizer.step()              # update weights

# ─── Save & Load ────────────────────────
torch.save(model.state_dict(), "model.pth")
model.load_state_dict(torch.load("model.pth"))
model.eval()                          # inference mode
```

## LLM API Cheat Sheet (OpenAI-compatible)

```python
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ─── Basic Chat ───────────────────────────
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain RAG in 2 sentences."}
    ],
    temperature=0.7,
    max_tokens=200
)
answer = response.choices[0].message.content

# ─── Streaming ───────────────────────────
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a poem."}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")

# ─── Structured Output ────────────────────
response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Extract: name, age, city"}],
    response_format=Flashcard
)

# ─── Embeddings ───────────────────────────
embeddings = client.embeddings.create(
    model="text-embedding-3-small",
    input="The quick brown fox"
)
vector = embeddings.data[0].embedding  # 1536-dim list

# ─── Tool Calling ─────────────────────────
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's the weather?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}}
            }
        }
    }]
)
```

## RAG Cheat Sheet

```python
# ─── 1. Chunk Document ────────────────────
def chunk_text(text, size=500, overlap=100):
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunks.append(text[i:i+size])
    return chunks

# ─── 2. Embed ─────────────────────────────
def embed(text):
    return client.embeddings.create(
        model="text-embedding-3-small", input=text
    ).data[0].embedding

# ─── 3. Store in Vector DB ────────────────
import chromadb
db = chromadb.PersistentClient(path="./chroma")
collection = db.create_collection("docs")
for i, chunk in enumerate(chunks):
    collection.add(
        ids=[f"chunk_{i}"],
        embeddings=[embed(chunk)],
        documents=[chunk]
    )

# ─── 4. Retrieve ─────────────────────────
results = collection.query(
    query_embeddings=[embed(question)],
    n_results=3
)
context = "\n\n".join(results["documents"][0])

# ─── 5. Generate ─────────────────────────
prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
answer = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
).choices[0].message.content
```

## FastAPI Cheat Sheet

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI(title="My API")

# ─── Model ────────────────────────────────
class Item(BaseModel):
    name: str
    price: float
    tags: List[str] = []

# ─── Endpoints ────────────────────────────
@app.get("/")
def root():
    return {"message": "Hello"}

@app.get("/items/{item_id}")
def get_item(item_id: int):
    if item_id not in items:
        raise HTTPException(404, "Not found")
    return items[item_id]

@app.post("/items")
def create_item(item: Item):
    items.append(item)
    return item

@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    items[item_id] = item
    return item

@app.delete("/items/{item_id}")
def delete_item(item_id: int):
    del items[item_id]
    return {"deleted": item_id}

# ─── Run ──────────────────────────────────
# uvicorn main:app --reload --port 8000
# Docs at http://localhost:8000/docs
```

## Docker Cheat Sheet

```dockerfile
# Dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# ─── Commands ────────────────────────────
docker build -t myapp .           # build image
docker run -p 8000:8000 myapp     # run container
docker ps                         # list running
docker logs <container>           # view logs
docker stop <container>           # stop
docker exec -it <container> bash  # shell into container

# ─── Docker Compose ──────────────────────
# docker-compose.yml:
# version: "3.9"
# services:
#   app:
#     build: .
#     ports: ["8000:8000"]
#     environment:
#       - OPENAI_API_KEY=${OPENAI_API_KEY}
#   db:
#     image: postgres:15
#     environment:
#       POSTGRES_PASSWORD: secret
#     volumes: ["pgdata:/var/lib/postgresql/data"]
# volumes:
#   pgdata:

docker-compose up -d              # start
docker-compose down               # stop
docker-compose logs -f            # tail logs
```

## Git Cheat Sheet

```bash
# ─── Setup ───────────────────────────────
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
git clone https://github.com/user/repo.git

# ─── Daily Workflow ──────────────────────
git status                        # see what changed
git add .                         # stage all changes
git add specific_file.py          # stage one file
git commit -m "Add feature"       # commit
git push                          # push to remote
git pull                          # pull latest

# ─── Branching ───────────────────────────
git branch feature                # create branch
git checkout feature              # switch to it
git checkout -b feature           # create + switch
git merge feature                 # merge into current

# ─── Undo ───────────────────────────────
git checkout -- file              # discard changes
git reset HEAD file               # unstage
git revert <commit>               # undo a commit
```

## Math Quick Reference

```
LINEAR ALGEBRA
  Vector: [a, b, c] — ordered list of numbers
  Dot Product: a·b = Σ(aᵢ × bᵢ) — measures similarity
  Matrix Multiply: (A @ B)[i,j] = Σ A[i,k] × B[k,j]
  Transpose: Aᵀ — swap rows and columns
  Identity: I — 1s on diagonal, 0s elsewhere
  Inverse: A⁻¹ — A @ A⁻¹ = I

PROBABILITY
  P(A) = favorable / total
  P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
  P(A|B) = P(A ∩ B) / P(B)  — conditional
  Bayes: P(A|B) = P(B|A) × P(A) / P(B)
  Mean: μ = Σ(xᵢ) / n
  Variance: σ² = Σ(xᵢ - μ)² / n
  Std Dev: σ = √(σ²)

CALCULUS
  Derivative: rate of change — d/dx(xⁿ) = n·xⁿ⁻¹
  Chain Rule: d(f(g(x)))/dx = f'(g(x)) × g'(x)
  Partial Derivative: derivative w.r.t. one variable
  Gradient: vector of all partial derivatives
  Gradient Descent: w_new = w - lr × ∇L

LOSS FUNCTIONS
  MSE: L = (1/n) × Σ(yᵢ - ŷᵢ)²
  Cross-Entropy: L = -Σ(yᵢ × log(ŷᵢ))
  Binary CE: L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```
