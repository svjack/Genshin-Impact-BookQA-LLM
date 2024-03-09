<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">Genshin-Impact-BookQA-LLM</h3>

  <p align="center">
   		A Book Question Answer Project supported by LLM (build by LangChain Haystack ChatGLM Mistral OLlama)
    <br />
  </p>
</p>

[中文介绍](README.md)

## Brief introduction

### BackGround
[Genshin Impact](https://genshin.hoyoverse.com/en/) is an action role-playing game developed by miHoYo, published by miHoYo in mainland China and worldwide by Cognosphere, 
HoYoverse. The game features an anime-style open-world environment and an action-based battle system using elemental magic and character-switching. 

In the Game, some background settings are described by [Books](https://bbs.mihoyo.com/ys/obc/channel/map/189/68?bbs_presentation_style=no_header).
Let's take a shot.
<img src="imgs/book_shot.png" alt="Girl in a jacket" width="1050" height="950">

This project is an attempt to build Chinese Q&A on the different LLM support RAG system.

### Try Demo on the fly


|Name | HuggingFace Space link |
|---------|--------|
| Genshin Impact Book QA Haystack Demo 📈 | https://huggingface.co/spaces/svjack/genshin-impact-bookqa-haystack |

The Demo use Huggingface Inference Api that call [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) to perform Q&A tasks.
Because the base model is not fintuned in Chinese but have more better inference capabilities than most below 10B models. You can take this deploy version as a free preview version. 

<img src="imgs/haystack_demo.png" alt="Girl in a jacket" width="1050" height="300"> <br/><br/>

## Installation and Running
### Install and Running Step
In the concept, the project can be divided into two parts, Basic_Part and LLM_Part. <br/>
* <b>Basic_Part</b> contains modules: [LangChain](https://github.com/langchain-ai/langchain) [SetFit](https://github.com/huggingface/setfit) you should install all of them By <br/>
```bash
pip install -r basic_requirements.txt
```
* <b>LLM_Part</b> are modules that you should choose one to install: [HayStack](https://github.com/deepset-ai/haystack) [chatglm.cpp](https://github.com/li-plus/chatglm.cpp) [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) [ollama](https://github.com/ollama/ollama)<br/> <br/>

Below are different LLM Repo types with their install and running command
|LLM Repo Name | LLM Model Name | Install Command in Linux | Run Gradio Demo Command |
|---------|--------|--------|--------|
| HayStack | Mistral-7B (based on huggingface inference) | pip install -r basic_requirements.txt && pip install haystack-ai==2.0.0b5 | python haystack_bookqa_gradio.py |
| llama-cpp-python | Mistral-7B (based on llama-cpp) | pip install -r basic_requirements.txt && pip install llama-cpp-python==0.2.55 | python mistral_bookqa_gradio.py |
| chatglm.cpp | chatglm3-6b | pip install -r basic_requirements.txt && pip install chatglm-cpp==0.3.1 | python chatglm_bookqa_gradio.py |
| ollama | Qwen-7B | pip install -r basic_requirements.txt && wget https://ollama.com/install.sh && sh ./install.sh && pip install ollama==0.1.6 && sudo systemctl start ollama | python ollama_qwen7b_bookqa_gradio.py |

### Note
I recommand you run the demo on GPU (10GB gpu memory is enough) <br/><br/>

## Datasets and Models
### Datasets
|Name | Type | HuggingFace Dataset link |
|---------|--------|--------|
| svjack/genshin_book_chunks_with_qa_sp | Genshin Impact Book Content | https://huggingface.co/datasets/svjack/genshin_book_chunks_with_qa_sp |
| svjack/bge_small_book_chunks_prebuld | Genshin Impact Book Embedding | https://huggingface.co/datasets/svjack/bge_small_book_chunks_prebuld |

### Basic Models
|Name | Type | HuggingFace Model link |
|---------|--------|--------|
| svjack/bge-small-book-qa | Embedding model | https://huggingface.co/svjack/bge-small-book-qa |
| svjack/setfit_info_cls | Text Classifier | https://huggingface.co/svjack/setfit_info_cls |

### LLM Models
|Name | Type | HuggingFace Model link |
|---------|--------|--------|
| svjack/chatglm3-6b-bin | ChatGLM3-6B 4bit quantization | https://huggingface.co/svjack/chatglm3-6b-bin |
| svjack/mistral-7b | Mistral-7B 4bit quantization | https://huggingface.co/svjack/mistral-7b |

<br/><br/>

## Architecture
This project has a traditional RAG structure.<br/>
[svjack/bge-small-book-qa](https://huggingface.co/svjack/bge-small-book-qa) is a self-trained embedding model
for recall genshin book contents (split by langChain TextSplitter). [svjack/setfit_info_cls](https://huggingface.co/svjack/setfit_info_cls) is a self-trained text classifier for determine whether the content is relevant to the query. <br/> <br/>

LLM Part have 4 different llm frameworks: [HayStack](https://github.com/deepset-ai/haystack) [chatglm.cpp](https://github.com/li-plus/chatglm.cpp) [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) [ollama](https://github.com/ollama/ollama) one can choose to answer the query based on the content recalled by embedding and filter out by text classifier.<br/> 

### Note
[HayStack](https://github.com/deepset-ai/haystack) [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) and [ollama](https://github.com/ollama/ollama) are repos contains many different llms. You can try to use different llms and change the model name or model 
file in the gradio scripts.<br/> * For the ability of understanding the query and context, i recommand you use Mistral-7B in Huggingface Inference Api and Intel/neural-chat in ollama. <br/> * For the ability of answer quality in Chinese, i recommand you Qwen-7B in ollama and ChatGLM3-6B in chatglm.cpp. 







