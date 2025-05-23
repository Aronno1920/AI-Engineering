AI Engineering Bootcamp for Programmers


# Study Plan | ১৯ Modules


## WEEK 0 | শুরুর আগে শুরু
59 recorded video on Python
--- 
## WEEK 1 | APIs, File Handling, and Data Manipulation
Objective: Enable students to collect and manipulate real-world data.

Live Class 1: What is a CSV file | Reading CSV using csv.reader() | Reading rows as dictionary using csv.DictReader() | What is JSON | Reading JSON file using json.load() | Converting JSON string using json.loads() | File modes: r, w, a | Reading text file using open() | Writing text file using write() | Using with statement for safe file handling

Live Class 2: API Requests with requests module | Making GET and POST requests | Handling API response data | Basics of web scraping | Using requests and BeautifulSoup | Extracting data from HTML elements | Introduction to data cleaning | Handling missing values with Pandas | Removing duplicates and irrelevant columns | Formatting and transforming data types
---
## WEEK 2 | Math for Machine Learning
Objective: Build math foundations for machine learning models.

Live Class 1: Introduction to vectors | Vector addition and scalar multiplication | Understanding matrices | Matrix dimensions and structure | Matrix operations: addition and multiplication | Dot product between vectors | Geometric interpretation of dot product | Applications of dot product in data science

Live Class 2: Introduction to probability concepts | Basic rules of probability | What is statistics and its role in data analysis | Calculating mean and median | Understanding variance and standard deviation | Introduction to data distributions | Normal distribution and its properties | Other common distributions: binomial, Poisson
---
## WEEK 3 | Development Tools and Best Practices
Objective: Introduce Git, Jupyter, Colab, and environment setup.

Live Class 1: What is Git and why use it | Basic Git commands: init, add, commit | Understanding version control | What is GitHub and how it works | Creating and managing repositories on GitHub | Cloning and pushing code | Branching and merging basics | Typical project folder structure | Best practices for organizing codebase

Live Class 2: What is Jupyter Notebook | Running and managing code cells | Introduction to Google Colab | Uploading and sharing notebooks | Writing clean and readable code | Using comments and proper indentation | Writing Markdown in notebooks | Formatting text, lists, and code blocks with Markdown | Creating headings and notes for better understanding
---
## WEEK 4 | Regression Models and Evaluation
Live Class 1: Introduction to linear regression | Fitting a line to data using least squares | Understanding slope and intercept | What is overfitting in regression | Introduction to regularization | Ridge regression and L2 penalty | Lasso regression and L1 penalty | Comparing ridge and lasso | When to use ridge vs lasso

Live Class 2: Introduction to model evaluation | What is Mean Absolute Error (MAE) | Interpreting MAE in regression models | What is Root Mean Squared Error (RMSE) | Difference between MAE and RMSE | What is R-squared (R²) | Interpreting R² as model accuracy | Choosing the right metric for evaluation

Project: House Price Predictor
---


## WEEK 5 | Classification Models and Evaluation
Live Class 1: Introduction to logistic regression | Understanding sigmoid function and probabilities | Binary classification with logistic regression | Introduction to K-Nearest Neighbors (KNN) | How KNN makes predictions based on distance | Choosing the right value of K | Introduction to decision trees | Splitting criteria: Gini and Entropy | Overfitting and pruning in decision trees | Comparing logistic, KNN, and decision trees

Live Class 2: What is a confusion matrix | Understanding TP, FP, TN, FN | Calculating precision and recall | What is F1 score and why it matters | Balancing precision and recall | Introduction to ROC curve | Interpreting ROC-AUC score | Choosing metrics based on problem type

Project: Diabetes Detection System
---

## WEEK 6 | Unsupervised Learning and Dimensionality Reduction
Live Class 1: Introduction to K-Means clustering | How K-Means groups similar data points | Choosing number of clusters | What is the Elbow Method | Using Elbow Method to find optimal K | Introduction to PCA (Principal Component Analysis) | Reducing dimensionality with PCA | Interpreting principal components | PCA vs clustering: when to use what

Live Class 2: Introduction to customer segmentation | Importance of segmenting customers | Clustering customers based on behavior or demographics | Using K-Means for customer segmentation | Visualizing clusters with scatter plots | Interpreting the segmented data | Visualizing customer profiles using bar charts and histograms | Using heatmaps for correlation analysis

Project: Customer Segmentation Engine
---

## WEEK 7 | Introduction to Deep Learning
Live Class 1: Introduction to neural networks | Structure of a neural network: neurons, layers, activation functions | Forward propagation process | Calculating outputs in a neural network | What is backpropagation | Adjusting weights during backpropagation | Gradient descent and learning rate | Role of loss function in training | Training a neural network through epochs

Live Class 2: Introduction to activation functions | Common activation functions: Sigmoid, ReLU, Tanh | Why activation functions are needed in neural networks | Introduction to loss functions | Mean Squared Error and Cross-Entropy Loss | How loss functions guide model training | Implementing a simple neural network | Forward propagation in a simple NN | Using backpropagation to update weights | Training the model with gradient descent

Project: MNIST Digit Classifier
---

## WEEK 8 | Computer Vision with CNNs
Live Class 1: Introduction to Convolutional Neural Networks (CNN) | What is computer vision? Use cases in industry | Basics of CNNs: convolution, pooling, flatten, FC | CNN architecture overview: input, convolution, pooling, fully connected layers | What is a convolution layer and its purpose | Applying filters to input images | Understanding kernel size and stride | Max pooling and average pooling layers | Reducing spatial dimensions with pooling | Role of convolution and pooling in feature extraction

Live Class 2: Introduction to image classification | Overview of CNN for image classification | Using Keras for building CNN models | Loading and preprocessing image data | Defining the CNN architecture in Keras | Adding convolution, activation, and pooling layers | Flattening and adding fully connected layers | Compiling the model with optimizer, loss, and metrics | Training the model with image data | Evaluating model performance on test data | Image preprocessing techniques: resizing, normalization | Data augmentation for Image Data 

Project: Dog vs Cat Image Classifier
---

## WEEK 9 | Recurrent Neural Networks and Time-Series Forecasting
Live Class 1: Introduction to Recurrent Neural Networks (RNN) | How RNNs handle sequential data | Vanishing gradient problem in RNNs | Introduction to Long Short-Term Memory (LSTM) networks | How LSTMs address the vanishing gradient issue | Structure of LSTM cells: forget, input, and output gates | Using RNNs and LSTMs for time series and text data | Sequential data processing with RNNs and LSTMs | Applications of RNN and LSTM in NLP and time series analysis

Live Class 2: Introduction to time-series forecasting | Using LSTM for sequential data prediction | Preprocessing time-series data for LSTM | Reshaping data for LSTM input | Defining LSTM architecture for forecasting | Training the LSTM model on time-series data | Evaluating the model with loss functions | Making predictions using the trained LSTM model | Visualizing forecast results vs actual values | Fine-tuning the model for better performance

Project: Sales Prediction Tool
---

## WEEK 10 | Model Deployment & Monitoring with FastAPI, Docker and MLflo
Live Class 1: Building Scalable ML APIs with FastAPI
What is FastAPI and why it's ideal for ML deployment | Setting up a FastAPI app for serving predictions | Creating robust endpoints with Pydantic for validation | Serving ML models using joblib or pickle | Structuring APIs: /predict, /health, /info endpoints | Testing APIs using Swagger UI, ReDoc, and Postman | Intro to async handling and real-time performance advantages 

Live Class 2: Dockerizing Your ML Service & Intro to MLflow
Docker Focus: | What is Docker and why use it for ML APIs | Writing a Dockerfile for FastAPI apps | Building Docker images and running containers | Exposing containerized APIs over localhost/port | Testing containerized predictions with curl/Postman | Analyzing API responses and troubleshooting | Basics of cloud deployment | Deploying Dockerized applications on cloud platforms | Using services like AWS, Heroku, or Google Cloud for deployment

Project: Production Ready ML Model Pipeline with FastAPI, Docker & MLflow
---

## WEEK 11 | Natural Language Processing Fundamentals
Live Class 1: Introduction to text preprocessing | What is tokenization and how it works | Breaking text into words or sentences | Introduction to lemmatization | Reducing words to their base form using lemmatization | What are stopwords and why remove them | Identifying and filtering stopwords in text | Using libraries like NLTK or SpaCy for preprocessing | Combining tokenization, lemmatization, and stopwords removal for clean text data

Live Class 2: Introduction to word embeddings | What is TF-IDF (Term Frequency-Inverse Document Frequency) | How TF-IDF captures word importance in a corpus | Introduction to Word2Vec and its two models: Continuous Bag of Words (CBOW) and Skip-gram | Training Word2Vec on text data to create word embeddings | What is GloVe (Global Vectors for Word Representation) | How GloVe differs from Word2Vec and captures global word relationships | Using pre-trained embeddings for NLP tasks | Comparing TF-IDF, Word2Vec, and GloVe for various applications

Project: Sentiment Analysis on Product Reviews
---

## WEEK 12 | Transformers and Hugging Face Models
Live Class 1: Introduction to the attention mechanism | How attention allows models to focus on important parts of the input | Types of attention: hard and soft attention | Self-attention and its role in sequence processing | Introduction to Transformer architecture | Key components of Transformers: Encoder and Decoder | Multi-head attention mechanism | Positional encoding to preserve sequence order | Advantages of Transformers over RNNs and LSTMs | Applications of Transformers in NLP tasks like translation and summarization

Live Class 2: Introduction to Hugging Face Transformers | Overview of BERT and GPT models | Understanding pre-trained models and transfer learning | Fine-tuning BERT for text classification tasks | Preparing the dataset for fine-tuning | Training BERT with custom data using Hugging Face’s Trainer API | Fine-tuning GPT for text generation tasks | Adjusting hyperparameters during fine-tuning | Evaluating model performance on downstream tasks | Using pre-trained Hugging Face models for various NLP applications

Project: Text Classification Using Transformers
---

## WEEK 13 | Prompt Engineering and GPT APIs
Live Class 1: Introduction to prompt patterns in NLP | Designing effective prompts for different tasks | Few-shot learning and how it reduces data requirements | Understanding few-shot, one-shot, and zero-shot learning | Introduction to OpenAI API | Overview of GPT models and their capabilities | Making API requests to OpenAI for text generation and understanding tasks | Prompt engineering for better results | Fine-tuning models with custom prompts for specific tasks | Using OpenAI API in real-world applications like chatbots, summarization, and translation

Live Class 2: Introduction to function calling with GPT models | Using GPT for structured outputs like JSON, tables, or lists | Designing prompts for structured responses | Calling external functions or APIs from within a prompt | Handling structured data in response formats | Parsing and formatting structured outputs from GPT | Integrating GPT with external systems for dynamic function calls | Use cases of function calling in automation, data extraction, and decision-making tasks | Best practices for optimizing structured output quality

Project: Resume Generator Using GPT
---

## WEEK 14 | Introduction to LangChain
Live Class 1: Introduction to chains in NLP workflows | Combining multiple tasks or steps using chains | Using memory in AI models for context retention | Implementing memory to maintain state across interactions | Creating prompt templates for consistent input formatting | Benefits of reusable prompt templates in large-scale applications | How chains and memory work together in task automation | Example use cases of chains, memory, and templates in complex tasks like conversation bots and multi-step reasoning | Optimizing performance with chains and templates


Live Class 2: Introduction to CSV/PDF Q&A pipelines | Extracting data from CSV files for Q&A tasks | Parsing PDF files for text-based question answering | Using OCR for extracting text from scanned PDFs | Building a pipeline to answer questions from structured or unstructured documents | Document indexing for fast retrieval | How to index documents for efficient search and retrieval

Project: Chatbot for Document-Based QA
---

## WEEK 15 | Building Intelligent Tool-Integrated Applications
Live Class 1: Introduction to LangChain Agent Executor | How the Agent Executor manages complex tasks | Using LangChain to build intelligent agents | Creating and integrating custom tools into LangChain | Leveraging external APIs and services as custom tools | Implementing intermediate memory for maintaining context across tasks | Storing temporary information to guide agent decision-making | Optimizing agent performance with efficient memory management | Example use cases of LangChain in conversational agents and workflow automation

Live Class 2: Introduction to API tools in LangChain | Using built-in API tools for integrating external services | Handling API requests and responses with LangChain | Introduction to file tools for managing document-based inputs | Reading, writing, and processing files in pipelines | Using LangChain to process and analyze file data (CSV, JSON, text) | Error handling strategies in LangChain workflows | Catching and managing exceptions in API and file tool interactions | Debugging and logging in LangChain | Best practices for building robust workflows with error handling

Project: Business Assistant Chatbot
---

## WEEK 16 | Multi-Agent Systems and Workflow Automation
Live Class 1: Introduction to CrewAI | How CrewAI enables automation and collaboration across tasks | Overview of AutoGen and its role in automating task generation | Using AutoGen for dynamic task creation and assignment | Task delegation strategies for optimizing workflow | Assigning tasks based on skills, priorities, and timelines | Managing task dependencies and ensuring smooth handoffs | Best practices for delegating tasks effectively within automated systems

Live Class 2: Introduction to multi-agent systems | How multiple agents interact and collaborate to solve problems | Example 1: Multi-agent system for recommendation engines | Agents working together to provide personalized suggestions | Example 2: Collaborative robots in manufacturing | Coordinating tasks between robots to improve efficiency | Example 3: Autonomous vehicle fleet management | Multiple agents managing traffic flow and navigation

Project: Research Planning and Summarization Assistant
---

## WEEK 17 | Retrieval-Augmented Generation and Vector Databases
Live Class 1: Introduction to FAISS (Facebook AI Similarity Search) | Using FAISS for efficient similarity search and nearest neighbor search | How FAISS works with high-dimensional data | Introduction to Chroma for scalable vector storage | Using Chroma for real-time, fast retrieval of vector-based data | Overview of LangChain retriever | Integrating LangChain with FAISS and Chroma for document retrieval | How LangChain handles retrieval-augmented generation (RAG) tasks | Benefits of combining FAISS, Chroma, and LangChain for high-performance information retrieval | Use cases in document search, semantic search, and knowledge extraction

Live Class 2: Introduction to the full Retrieval-Augmented Generation (RAG) pipeline | Ingestion of data for RAG systems: collecting and processing raw data | Preprocessing data for embedding generation | Using embeddings to represent data in vector space | Introduction to embeddings: how they capture semantic meaning | Generating embeddings with models like BERT or Sentence Transformers | Storing and indexing embeddings for efficient retrieval | Retrieval process: searching for relevant documents or data based on query

Project: Knowledge Base Chatbot with Vector DB
---

## WEEK 18 | Capstone Presentation and Career Support
