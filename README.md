# 🎀 CatNeural

Projeto acadêmico de Redes Neurais + MongoDB + Frontend.

A aplicação recebe uma imagem, usa uma rede neural para estimar a probabilidade de a imagem conter um gatinho, salva a análise no MongoDB e apresenta o resultado em uma interface web rosa e kawaii.

## Stack

- Frontend: React + Vite
- Backend: Python + FastAPI
- IA: TensorFlow / Keras
- Banco: MongoDB
- Estilo: CSS puro, responsivo
- Modelo: CNN com transfer learning (MobileNetV2)

## Estrutura

```text
catneural/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── .env.example
│   └── model/
│       └── train.py
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       └── styles.css
└── README.md
```

## 1. MongoDB

Crie um banco chamado `catneural`.

No `.env`:

```env
MONGO_URI=mongodb://localhost:27017
DB_NAME=catneural
```

MongoDB Atlas também pode ser usado.

## 2. Backend

```bash
cd backend
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Instale:

```bash
pip install -r requirements.txt
```

Execute:

```bash
uvicorn main:app --reload
```

API: `http://localhost:8000`

## 3. Modelo

O treinamento espera esta estrutura:

```text
dataset/
├── train/
│   ├── cat/
│   └── not_cat/
└── validation/
    ├── cat/
    └── not_cat/
```

Coloque imagens de gatos em `cat/` e imagens que não sejam gatos em `not_cat/`.

Depois:

```bash
python backend/model/train.py
```

O arquivo `catneural.keras` será criado em `backend/model/`.

Para um projeto acadêmico, registre no relatório a origem do dataset, quantidade de imagens e divisão treino/validação.

## 4. Frontend

```bash
cd frontend
npm install
npm run dev
```

Abra a URL mostrada pelo Vite.

Se o backend estiver em outra URL, altere `API_URL` em `src/App.jsx`.

## Fluxo

```text
Imagem
   ↓
React
   ↓
FastAPI
   ↓
MobileNetV2 + camada de classificação
   ↓
Probabilidade de gato
   ↓
MongoDB
   ↓
Dashboard
```

## Observação

O projeto foi estruturado para uma classificação binária: `cat` x `not_cat`. A porcentagem exibida deve ser interpretada como a confiança estimada pelo modelo, não como uma verdade absoluta.


## MongoDB rápido com Docker

Se você tiver Docker instalado:

```bash
docker compose up -d
```

Isso sobe o MongoDB na porta `27017` sem precisar instalar o banco localmente.

## Dataset

As pastas de dataset já estão criadas. Adicione suas imagens em:

```text
backend/dataset/train/cat/
backend/dataset/train/not_cat/
backend/dataset/validation/cat/
backend/dataset/validation/not_cat/
```

Para uma demonstração acadêmica, mantenha uma quantidade equilibrada de imagens nas duas classes e explique no relatório como o dataset foi obtido e dividido.

## Demonstração

Depois de treinar o modelo e iniciar a API, o fluxo da apresentação pode ser:

1. Abrir o Dashboard CatNeural.
2. Fazer upload de uma foto de gato.
3. Mostrar a previsão em porcentagem.
4. Mostrar que a análise foi persistida no MongoDB.
5. Atualizar o histórico.
6. Explicar a arquitetura CNN/transfer learning.
