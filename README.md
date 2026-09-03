# CatNeural

Aplicação acadêmica para classificação de imagens com inteligência artificial. O CatNeural recebe uma imagem, estima a probabilidade de ela conter um gato, persiste a análise no MongoDB e exibe o resultado em uma interface web responsiva.

> **Status: Em andamento**
>
> O projeto está em desenvolvimento contínuo. Funcionalidades, modelo e interface ainda podem receber melhorias antes da versão final.

## Visão geral

O projeto combina uma API em Python, um frontend em React e um modelo de visão computacional baseado em redes neurais convolucionais. A classificação utiliza transfer learning com MobileNetV2 para diferenciar as classes `cat` e `not_cat`.

## Tecnologias

| Camada | Tecnologia |
| --- | --- |
| Frontend | React, Vite e CSS responsivo |
| Backend | Python e FastAPI |
| Inteligência artificial | TensorFlow, Keras e MobileNetV2 |
| Banco de dados | MongoDB |

## Estrutura do projeto

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

## Pré-requisitos

Antes de iniciar, certifique-se de ter instalado:

- Python 3.10 ou superior;
- Node.js e npm;
- MongoDB local ou uma instância no MongoDB Atlas;
- Docker, caso prefira executar o MongoDB em um container.

## Configuração do MongoDB

Por padrão, a aplicação utiliza um banco chamado `catneural`. Configure as variáveis de ambiente no arquivo `.env` do backend:

```env
MONGO_URI=mongodb://localhost:27017
DB_NAME=catneural
```

O MongoDB Atlas também pode ser utilizado. Nesse caso, substitua `MONGO_URI` pela string de conexão fornecida pelo serviço.

### Usando Docker

Se o Docker estiver instalado, inicie o MongoDB com:

```bash
docker compose up -d
```

Esse comando disponibiliza o banco na porta `27017`.

## Execução do backend

```bash
cd backend
python -m venv .venv
```

Ative o ambiente virtual:

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Instale as dependências e inicie a API:

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

A API estará disponível em `http://localhost:8000`.

## Treinamento do modelo

O treinamento espera o dataset organizado da seguinte forma:

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

Execute o treinamento a partir da raiz do projeto:

```bash
python backend/model/train.py
```

Ao final, o arquivo `catneural.keras` será gerado em `backend/model/`.

Para fins acadêmicos, registre no relatório a origem do dataset, a quantidade de imagens e a divisão entre treino e validação.

## Execução do frontend

```bash
cd frontend
npm install
npm run dev
```

Abra no navegador a URL exibida pelo Vite.

Se o backend estiver em outra URL, atualize a constante `API_URL` em `frontend/src/App.jsx`.

## Fluxo da aplicação

```text
Imagem enviada
   |
   v
Frontend React
   |
   v
API FastAPI
   |
   v
MobileNetV2 + camada de classificacao
   |
   v
Probabilidade de gato
   |
   v
MongoDB e historico
   |
   v
Dashboard
```

## Interpretação dos resultados

O projeto realiza uma classificação binária entre `cat` e `not_cat`. A porcentagem apresentada representa a confiança estimada pelo modelo e não deve ser interpretada como uma verdade absoluta.

## Dataset

Adicione as imagens nas pastas correspondentes:

```text
backend/dataset/train/cat/
backend/dataset/train/not_cat/
backend/dataset/validation/cat/
backend/dataset/validation/not_cat/
```

Para uma demonstração acadêmica, mantenha uma quantidade equilibrada de imagens entre as duas classes. A procedência e a divisão do dataset devem ser documentadas no relatório do projeto.

## Demonstração

Depois de treinar o modelo e iniciar o MongoDB, a API e o frontend, o fluxo sugerido para uma apresentação é:

1. Abrir o Dashboard CatNeural.
2. Enviar uma imagem de gato ou de outra categoria.
3. Conferir a previsão em porcentagem.
4. Verificar a persistência da análise no MongoDB.
5. Consultar o histórico de análises.
6. Explicar a arquitetura CNN e o uso de transfer learning.

## Próximos passos

Como o desenvolvimento ainda está em andamento, os próximos aprimoramentos podem incluir:

- melhoria da qualidade e da diversidade do dataset;
- avaliação mais detalhada do desempenho do modelo;
- aprimoramento do tratamento de erros e da validação de imagens;
- evolução da interface e do histórico de análises;
- criação de testes automatizados para frontend e backend.

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
