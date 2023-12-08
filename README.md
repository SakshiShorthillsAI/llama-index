# llama-index
What is llama index 

LlamaIndex is a data framework for LLM-based applications to ingest, structure, and access private or domain-specific data. It’s available in Python (these docs) and Typescript.  

LlamaIndex is a user-friendly, flexible data framework connecting private, customized data sources to your large language models (LLMs). we can think of LlamaIndex as a managed interaction between you and an LLM. LlamaIndex takes some input data you provide and builds an index around it. Then it uses that index to help answer any questions related to the input data. LlamaIndex can build many types of indexes depending on the task at hand. You can use it to build a vector index, a tree index, a list index, or a keyword index. 

 

 

Why LlamaIndex? 

LLMs offer a natural language interface between humans and data. Widely available models come pre-trained on huge amounts of publicly available data like Wikipedia, mailing lists, textbooks, source code and more. 

However, while LLMs are trained on a great deal of data, they are not trained on your data, which may be private or specific to the problem you’re trying to solve. It’s behind APIs, in SQL databases, or trapped in PDFs and slide decks. 

LlamaIndex solves this problem by connecting to these data sources and adding your data to the data LLMs already have. This is often called Retrieval-Augmented Generation (RAG). RAG enables you to use LLMs to query your data, transform it, and generate new insights. You can ask questions about your data, create chatbots, build semi-autonomous agents, and more. To learn more, check out our Use Cases on the left. 

How can LlamaIndex help? 

LlamaIndex provides the following tools: 

Data connectors ingest your existing data from their native source and format. These could be APIs, PDFs, SQL, and (much) more. 

Data indexes structure your data in intermediate representations that are easy and performant for LLMs to consume. 

Engines provide natural language access to your data. For example: - Query engines are powerful retrieval interfaces for knowledge-augmented output. - Chat engines are conversational interfaces for multi-message, “back and forth” interactions with your data. 

Data agents are LLM-powered knowledge workers augmented by tools, from simple helper functions to API integrations and more. 

Application integrations tie LlamaIndex back into the rest of your ecosystem. This could be LangChain, Flask, Docker, ChatGPT, or… anything else! 

 

 

Installation and Setup 

Installation from Pip 

Install from pip: 

pip install llama-index 

Installation from Source 

Git clone this repository: git clone https://github.com/jerryjliu/llama_index.git. Then do the following: 

Install poetry - this will help you manage package dependencies 

poetry shell - this command creates a virtual environment, which keeps installed packages contained to this project 

poetry install - this will install the core package requirements 

(Optional) poetry install --with dev,docs - this will install all dependencies needed for most local development 

 
 

 

 The Indexes in LlamaIndex 

Let’s take a closer look at the indices you can build with LlamaIndex, how they work, and what sorts of use cases would best fit each. At their core, all of the index types in LlamaIndex are made up of “nodes”. Nodes are objects that represent a chunk of text from a document. 

List Index 

 

List index  

As the name suggests, a list index is an index that is represented like a list. First, the input data gets chunked into nodes. Then, the nodes are lined up sequentially. The nodes are also queried sequentially if no additional parameters are specified at query time. In addition to basic sequential querying, we can query the nodes using keywords or embeddings. 

 

 

Vector Store Index 

 

Vector Store Index  

The next index that LlamaIndex offers is a vector store index. This type of index stores the nodes as vector embeddings. LlamaIndex offers a way to store these vector embeddings locally or with a purpose-built vector database like Milvus. When queried, LlamaIndex finds the top_k most similar nodes and returns that to the response synthesizer. 

Using a vector store index lets you introduce similarity into your LLM application. This is the best index for when your workflow compares texts for semantic similarity via vector search. For example, if you want to ask questions about a specific type of open-source software, you would use a vector store index. 

 

 

Tree Index  

 

 

A tree index builds a tree out of your input data. The tree index is built bottom-up from the leaf nodes, the original input data chunks. Each parent node is a summary of the leaf nodes. LlamaIndex uses GPT to summarize the nodes to build the tree. Then when building a response to a query, the tree index can traverse down through the root node to the leaf nodes or build directly from chosen leaf nodes. 

The tree index offers a more efficient way to query long chunks of text. It can also be used to extract information from different parts of the text. Unlike the list index, a tree index does not need to be queried sequentially. 

Keyword Index 

 

Finally we come to the keyword index. The keyword index is a map of keywords to nodes that contain those keywords. This is a many-to-many mapping. Each keyword may point to multiple nodes and each node may have multiple keywords that map to it. During query time, keywords are extracted from the query and only the mapped nodes are queried. 

The keyword index provides a more efficient way to query large amounts of data for certain keywords. This is most useful when you know what the user is querying for. For example, if you’re querying through health care related documents and you only care about documents related to COVID 19. 

 

 

 

The workflow of LlamaIndex 

 

 

 

 

The workflow of LlamaIndex can be broken down into two primary aspects: data processing and querying. 

Data processing 

 

In the data processing phase, LlamaIndex partitions your knowledge base (for example, organizational documents) into chunks stored as ‘node’ objects. These nodes collectively form an ‘index’ or a graph. The process of ‘chunking’ is crucial as LLMs have a limited input token capacity, making it necessary to devise a strategy to process large documents smoothly and continuously. 

The index graph could have different configurations, such as a simple list structure, a tree structure, or a keyword table. Moreover, indexes can also be composed of different indexes instead of nodes. For instance, you can create separate list indexes over Confluence, Google Docs, and emails and create an overarching tree index over these list indexes. 

When it comes to creating nodes, LlamaIndex uses ‘textSplitter’ classes which break up the input to an LLM to stay within token limitations. However, you can create custom splitters or generate your chunks beforehand for more control over document chunking. 

 

Querying 

Querying an index graph involves two primary tasks. Initially, a collection of nodes relevant to the query are fetched. A response_synthesis module is utilized, utilizing these nodes and the original query to generate a logical response. The relevance of a node is determined based on the index type. Let’s review how these relevant nodes are procured in different setups: 

    List index querying: A list index sequentially employs all the nodes in the list to generate a response. The query, accompanied by information from the first node, is dispatched to the LLM as a prompt. The prompt might be structured like “given this {context}, answer this {query},” where the node supplies the context and the query is the original query. The LLM’s returned response is refined as we progress through the nodes. The current response, query, and the next node are embedded in a prompt resembling “given the response so far {current_response}, and the following context: {context}, refine the response to the query {query} in line with the context.” This process continues until all nodes have been traversed. By default, this index retrieves and transfers all nodes in an index to the response synthesis module. However, when the query_mode parameter is set to “embedding,” only the nodes with the highest similarity (measured by vector similarities) are fetched for response_synthesis. 

    Vector index querying: A vector index calculates embeddings for each document node and stores them in a vector database like PineCone or Vertex AI matching engine. The key difference in retrieval compared to the list index is that only nodes surpassing a specific relevance threshold to the query are fetched and delivered to the response_synthesis model. 

    Response synthesis: This module offers several methods to create the response. 

    Create and refine: This is the standard mode for a list index. The list of nodes is sequentially traversed, and at each step, the query, the response so far, and the current node’s context are embedded in a prompt template that prompts the LLM to refine the query response following the new information in the current node. 

    Tree summarize: This is similar to the tree index in that a tree is created from the chosen candidate nodes. However, the summarization prompt used to derive the parent nodes is seeded with the query. Moreover, the tree construction continues until a single root node is reached, which contains the query’s answer, composed of the information in all the selected nodes. 

    Compact: This method is designed to be cost-effective. The response synthesizer is instructed to cram as many nodes as possible into the prompt before reaching the LLM’s token limitation. Suppose too many nodes exist to fit into one prompt. In that case, the synthesizer will perform this in stages, inserting the maximum possible number of nodes into the prompt at each step and refining the answer in subsequent steps. It’s worth noting that the prompts used to interact with the LLMs can be customized. For example, you can seed tree construction with your own custom summary prompt. 

    Composability: A useful feature of LlamaIndex is its ability to compose an index from other indexes rather than nodes. Suppose you need to search or summarize multiple heterogeneous data sources. You can create a separate index over each data source and a list index over these indexes. A list index is suitable because it generates and refines an answer (whether it’s a summary or query answer) iteratively by stepping through each index. You need to register a summary for each lower-level index. This is because, like other modules and classes, this feature relies on prompting LLMs to, for instance, identify the relevant sub-indexes. In a tree index with a branching factor of 1, this summary is used to identify the correct document to direct the query. 

 

Storage 

Storage is a critical component of this library for developers, necessitating space for vectors (representing document embeddings), nodes (representing chunks of documents), and the index itself. By default, nearly everything is stored in memory, except for vector store services such as PineCone, which house your vectors in their databases. To ensure the persistence of the information, these in-memory storage objects can be saved to disk for future reloading. 

Looking into the available options for storage, let’s discuss them one by one: 

    Document stores: MongoDB is the sole external alternative to in-memory storage. Specifically, two classes, MongoDocumentStore and SimpleDocumentStore, manage the storage of your document nodes either in a MongoDB server or in memory. 

    Index stores: Similar to document stores, MongoIndexStore and SimpleIndexStore, two classes, handle the storage of index metadata in either MongoDB or memory. 

    Vector stores: Besides the SimpleVectorStore class that keeps your vectors in memory, LlamaIndex supports a wide range of vector databases akin to LangChain. It’s crucial to note that while some vector databases house both documents and vectors, others, like PineCone, exclusively store vectors. Nonetheless, hosted databases like PineCone allow for highly efficient complex computations on these vectors compared to in-memory databases like Chroma. 

    Storage context: Once you have configured your storage objects to fit your needs or left them as default settings, a storage_context object can be created from them. This allows your indexes to account for everything comprehensively. 

 

RAG in llama-index 

The basic usage pattern for LlamaIndex is a 5-step process that takes you from your raw, unstructed data to LLM generated content based on that data: 

    Load documents 

    Parse Documents into Nodes 

    Build an Index 

    Query the index 

    Parse the response 

 

 

LOAD DOCUEMENTS 

To start with, the initial step is to load data in the form of Document objects. 

For this purpose, LlamaIndex has several data loaders which will help you load Documents via the load_data method. 

 

To use excel loader in llama-index - Pandas Excel Loader 

This loader extracts the text from a column of a local .xlsx file using the pandas Python package. A single local file is passed in each time you call load_data. 

To use this loader, you need to pass in a Path to a local file, along with a sheet_name from which sheet to extract data. The default sheet_name=None, which means it will load all the sheets in the excel file. You can set sheet_name="Data1 to load only the sheet named "Data1". Or you can set sheet_name=0 to load the first sheet in the excel file. You can pass any additional pandas configuration options to the pandas_config parameter. 

from pathlib import Path 
from llama_index import download_loader 
 
PandasExcelReader = download_loader("PandasExcelReader") 
 
loader = PandasExcelReader(pandas_config={"header": 0}) 
documents = loader.load_data(file=Path('./data.xlsx')) 

 

To use loader in json - JSON Loader 

This loader extracts the text in a formatted manner from a JSON or JSONL file. A single local file is passed in each time you call load_data. 

Usage 

To use this loader, you need to pass in a Path to a local file and set the is_jsonl parameter to True for JSONL files or False for regular JSON files. 

JSON 

from pathlib import Path 
from llama_index import download_loader 
 
JSONReader = download_loader("JSONReader") 
 
loader = JSONReader() 
documents = loader.load_data(Path('./data.json'))  

 

 

 

 

 

To use loader to load csv in llama-index - Simple CSV Loader 

This loader extracts the text from a local .csv file by directly reading the file row by row. A single local file is passed in each time you call load_data. 

Usage 

To use this loader, you need to pass in a Path to a local file. 

from pathlib import Path 
from llama_index import download_loader 
 
SimpleCSVReader = download_loader("SimpleCSVReader") 
 
loader = SimpleCSVReader(encoding="utf-8") 
documents = loader.load_data(file=Path('./transactions.csv'))  

To load using SimpleDirectory Reader 

The easiest reader to use is our SimpleDirectoryReader, which creates documents out of every file in a given directory. It is built in to LlamaIndex and can read a variety of formats including Markdown, PDFs, Word documents, PowerPoint decks, images, audio and video. 

from llama_index import SimpleDirectoryReader 
 
documents = SimpleDirectoryReader("./data").load_data()  

 

 

 

What are Node objects in LlamaIndex? 

A Node object in LlamaIndex represents a “chunk” or a portion of a source Document. 

This could be a text chunk, an image, or other types of data. Similar to Documents, Nodes also contain metadata and relationship information with other nodes. 

Nodes are considered a first-class citizen in LlamaIndex. 

This means you can define Nodes and all its attributes directly. 

Alternatively, you can also “parse” source Documents into Nodes using the NodeParser classes. By default, every Node derived from a Document will inherit the same metadata from that Document. For example, a “file_name” field in the Document is propagated to every Node. 

 

Our Chunking Strategy 

You’ll make use of a strategy that utilized smaller child chunks that refer to bigger parent chunks. 

To do this you’ll use first use a SimpleNodeParser with a SentenceSplitter to create “base nodes” which are larger chunks of text. 

Then, you’ll create child nodes using a SentenceWindowNodeParser, which will have nodes that represent bullet points from the slide deck and metadata that references a “window” of a few bullets on either side. 

 

SimpleNodeParser 

SimpleNodeParser converts documents into a list of nodes. It offers flexibility in how the document is parsed, allowing for customization in terms of chunk size, overlap, and inclusion of metadata. 

The SimpleNodeParser with a SentenceSplitter is used when you want to break down your documents into chunks of a specific size, with each chunk being a node. 

This is particularly useful when dealing with large documents that need to be divided into manageable pieces for processing. 

 

SentenceSplitter 

The SentenceSplitter is a type of text splitter that breaks down the text into sentences. 

This is useful when you want to maintain the integrity of individual sentences within each chunk. 

 

SentenceWindowNodeParser 

The SentenceWindowNodeParser class is designed to parse documents into nodes (sentences) and capture a window of surrounding sentences for each node. 

This can be useful for context-aware text processing, where understanding the surrounding context of a sentence can provide valuable insights. 

 

Processing Nodes 

The code below processes a list of base nodes (slides_nodes). 

For each base node, it generates sub-nodes using the SentenceWindowNodeParser (with custom settings). Then, it converts the base nodes and their corresponding sub-nodes into IndexNode instances. 

The final list of IndexNode instances is stored in all_nodes. 


 

What is an IndexNode in LlamaIndex? 

An IndexNode is a node object used in LlamaIndex. 

It represents chunks of the original documents that are stored in an Index. The Index is a data structure that allows for quick retrieval of relevant context for a user query, which is fundamental for retrieval-augmented generation (RAG) use cases. 

At its core, the IndexNode inherits properties from a TextNode, meaning it primarily represents textual content. 

However, the distinguishing feature of an IndexNode is its index_id attribute. This index_id acts as a unique identifier or reference to another object, allowing the node to point or link to other entities within the system. 

This referencing capability adds a layer of connectivity and relational information on top of the textual content. 

For example, in the context of recursive retrieval and node references, smaller chunks (represented as IndexNode objects) can point to bigger parent chunks. Smaller chunks are retrieved during query time, but references to bigger chunks are followed. 

This allows for more context for synthesis. 

Conceptually, you can think of an IndexNode as a bridge or link node. 

While it holds its own textual content (inherited from TextNode), it also serves as a pointer or reference to another entity in the system, as indicated by its index_id. 

This dual nature allows for more complex and interconnected data structures, where nodes can represent both content and relationships to other objects. 

 

Embedding model and LLM 

For this example, you’ll use the BGE embedder from HuggingFace as our embeddings model. 

 

To use chromadb database 

    Set up ChromaVectorStore and load in data 

    How to Persist: Saving to Disk 

    Load from Disk 

 

Service Context 

The ServiceContext in LlamaIndex is a utility container that bundles commonly used resources during the indexing and querying stages of a LlamaIndex pipeline or application. 

It can be used to set both global and local configurations at specific parts of the pipeline 

VectorStoreIndex in LlamaIndex 

    A VectorStoreIndex in LlamaIndex is a type of index that uses vector representations of text for efficient retrieval of relevant context. 

    It is built on top of a VectorStore, which is a data structure that stores vectors and allows for quick nearest neighbor search. 

    The VectorStoreIndex takes in IndexNode objects, which represent chunks of the original documents. 

    It uses an embedding model (specified in the ServiceContext) to convert the text content of these nodes into vector representations. These vectors are then stored in the VectorStore. 

    During query time, the VectorStoreIndex can quickly retrieve the most relevant nodes for a given query. 

    It does this by converting the query into a vector using the same embedding model, and then performing a nearest neighbor search in the VectorStore. 

vector_index_chunk = VectorStoreIndex( 

all_nodes, service_context=service_context 

) 

The as_retriever method of a VectorStoreIndex in LlamaIndex is used to create a retriever object from the index. 

A retriever is a component that is responsible for fetching relevant context from the index given a user query. 

When you call as_retriever on a VectorStoreIndex, it returns a VectorStoreRetriever object. 

This retriever uses the vector representations stored in the VectorStoreIndex to perform efficient nearest neighbor search and retrieve the most relevant IndexNode objects for a given query. 

Below, this is configured to fetch the 2 most similar chunks. 

vector_retriever_chunk = vector_index_chunk.as_retriever(similarity_top_k=2) 

 

RecurseiveRetriever in LlamaIndex 

The RecursiveRetriever is designed to recursively explore links from nodes to other retrievers or query engines. 

This means that when the retriever fetches nodes, if any of those nodes point to another retriever or query engine, the RecursiveRetriever will follow that link and query the linked retriever or engine as well. 

If any of the retrieved nodes are of type IndexNodes, the retriever will specifically explore the linked retriever or query engine associated with those IndexNodes and initiate a query on that linked entity. 

RecursiveRetriever is designed to handle complex retrieval tasks, especially when data is spread across different retrievers or query engines. It follows links, retrieves data from linked sources, and can combine results from multiple sources into a single coherent response. 

 

RetrieverQueryEngine in LlamaIndex 

A RetrieverQueryEngine in LlamaIndex is a type of query engine that uses a retriever to fetch relevant context from an index given a user query. 

It is designed to work with retrievers, such as the VectorStoreRetriever created from a VectorStoreIndex. 

The RetrieverQueryEngine takes a retriever and a response synthesizer as inputs. The retriever is responsible for fetching relevant IndexNode objects from the index, while the response synthesizer is used to generate a natural language response based on the retrieved nodes and the user query. 

 

Response mode 

For this example, you’ll use the "compact" response mode. 

Compact combines text chunks into larger consolidated chunks that more fully utilize the available context window, then refine answers across them. 

Refer to the docs for full description of all the response mode 

 

Core query functions 

Three primary components – Index, Retriever, and Query Engine – form the backbone of the process for soliciting information from your data or documents: 

    The Index, or indices, in LlamaIndex, is a data structure that quickly fetches relevant information from external documents based on a user’s query. It works by dividing documents into text sections known as “Node” objects and building an index from these pieces. LlamaIndex is foundational for use cases involving the Retrieval Augmented Generation (RAG) of information. In general, indices are built from documents and then used to create Query Engines and Chat Engines. This sets up a question-answer and chat system over your data. To get more specific, indices store data as Node objects, representing sections of the original documents while offering a Retriever interface for additional configuration and automation. 

    The Retriever is a tool for extracting and gathering relevant information based on a user’s query. It possesses the flexibility to be developed atop Indices, yet it can also be established independently. It plays a crucial role in constructing Query Engines (and Chat Engines), enabling the retrieval of pertinent context. 

    The Query Engine built atop the Index and Retriever, provides a universal interface for querying data. The Index, Retriever, and Query Engine come in various forms to accommodate different needs. 

    Chat engine: A Chat Engine provides an advanced interface for engaging in dialogue with your data, allowing for ongoing conversation instead of a solitary question and response. Imagine ChatGPT, but enhanced with information from your knowledge base. In principle, it resembles a Query Engine with statefulness, capable of maintaining a record of the conversation history. Consequently, it can respond by considering the context of past interactions 

 

 

 

 

 

LlamaIndex vs. LangChain  

LangChain is a framework centered around LLMs that offers a wide range of applications, such as chatbots, Generative Question-Answering (GQA), summarization, and more. The fundamental concept behind this framework is the ability to chain together various components, and enable the creation of sophisticated functionalities and use cases based on LLMs. These chains may consist of multiple components from several modules, including LLMs, memory, agents, and prompt templates. 

While there may be some overlap, LllamaIndex doesn’t utilize agents/chatbots/prompt management/etc., rather, it focuses on going deep into indexing and efficient information retrieval for LLM's. LlamaIndex can be used as a retrieval/tool in LangChain and provides tool abstractions so you can use a LlamaIndex query engine along with a LangChain agent. It also allows you to use any data loader within the LlamaIndex core repo or in LlamaHub as an “on-demand” data query Tool within a LangChain agent. 

 

 

 


 
