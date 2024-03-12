<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">Genshin-Impact-BookQA-LLM</h3>

  <p align="center">
   		A Genshin Impact Book Question Answer Project supported by LLM (build by LangChain Haystack ChatGLM Mistral OLlama)
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

## Installation and Running Results
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

### Running Results



https://github.com/svjack/Genshin-Impact-BookQA-LLM/assets/27874014/dd90d1ee-669a-4a12-ab4e-86069064e0a4



https://github.com/svjack/Genshin-Impact-BookQA-LLM/assets/27874014/3c915195-952e-4289-bc88-1e44855c5582




https://github.com/svjack/Genshin-Impact-BookQA-LLM/assets/27874014/99f2ff32-f89e-46c2-8e8e-27a6cda85809



https://github.com/svjack/Genshin-Impact-BookQA-LLM/assets/27874014/b83c2764-fb49-4574-8ac9-be28ff1cf1f1



<br/>

Following are results of two problem in different models.

|    | Question                       | HayStack                                                                                                                                                                                                                                                                                                                                                                                                 | Mistral                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | ChatGLM-6B                                                                                                                                                                                                                                                                                                                                                               | QWen-7B                                                                                                                                                                                                                                                      |
|---:|:---------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  1 | 岩王帝君是一个什么样的人？ | 岩王帝君是一位古老的神明，在古代中被广泛敬仰和信仰。他是公正、决断、冷酷无私的神明，处事公正，决断无情，但缺乏凡人应有的感情。他的法度能保证交易公平、生活安全有序，同时从信仰中强化着自己的存在和力量。但即使是神，也无法左右凡人的信仰和质疑。即使是公道的守护神，也无法将规则钉入每个人心底。在某个女人来到他的工坊要求雕刻真实的岩王肖像的条件下，他闭门雕琢了三日，最终成形了一方玉牌上的岩王肖像。 | 岩王帝君是一位古老而强大的神明之一，在璃月地区拥有重大影响力。他曾经有过仙兽作战，战斗结束后，仙兽失去了在神战中守护凡人的意义，便选择隐居而生。有些仙兽仍然渴望守护璃月，向岩王帝君祈求将肉身化作永恒的磐岩，被允许成实。在璃月历史中，霓裳花被视为岩王帝君的象征之一，同时也是璃月人日常生活中常见到的物品之一，用于加工绸段和制香膏等用途。在璃月历史中也有传说，有一块星星坠落造成了璃月地区的巨渊和美玉金石，而一小块星碎在群岩中落下，被称为「遁玉陵」，这个古老遗迹中有许多古老残垣，传说名字「遁玉」意味着「美玉遁逃之处」。在璃月港西北方有一位名叫玉匠的人，他喜欢以令人意想不到的方式完成委托，如果客人要求他雕刻一张岩神肖像，他可能会雕刻出一只野猪或者其他奇怪形状。玉匠曾经遇到了一位女人，她要求他雕刻出真实的岩神肖像，否则她不会付钱。玉匠约定了三日之内完成任务。第一天玉匠与好友宴饮阔论，拒绝任何委托；第二天玉匠出门登山访玉；第三天玉匠才开始闭门雕琢璞玉，直到深夜才完成任务。 | 岩王帝君是一个神秘的人物，他的形象在无数的故事传说中得以展现。他订立规则与契约，却不强迫凡人以权威的完美模板生活。他深知规则与契约仅是手段而非目的，真正的平衡与隽永在于每一个人的觉醒与抉择。他有着深不可测的力量和智慧，同时也是一个善良和宽容的人。虽然关于岩王帝君的具体形象和故事很多，但我们无法确定他的真实身份和特点。只能从这些传说中了解到他的一些特点和品质。 | 岩王帝君不仅是提瓦特大地的统治者，也被他的子民纪念、崇拜，甚至爱戴。这表明他可能有着慈祥和被尊敬的一面，同时也显示了他的威严和统治地位。                                                                                                                     |
| 2 | 灵光颂的内容是什么样的？   | 灵光颂的内容是一种诗歌，它描述了那伽朱那呀（Bodhisattva Avalokitesvara）得到万种灵光的帮助，并使用这些灵光保护大地、洁净人们、开辟道路给善灵、滋养花海的善灵等。它们无论是在悲怨的河流中逆渡，还是面对天外漆黑的阴霾，都能保护住自己。这种灵光永恒地指引人们上升道路。                                                                                                                                   | 这首诗称为"灵光颂"，内容描述了一位人类或是其他实体，得到了万种灵光的帮助，已经超越了死亡，洁净了高天的渊薮，洗净了无罪之人的罪孽，在大地中蕴藏了一切业种。这位实体在世间的荣耀只是种子现行的幻象，但它已经逆渡悲怨的河流，渴饮甘露，为善灵开辟道路，使智慧的慈光普照四方。它得到了万种灵光的保护，滋养着花海的善灵，它们在渴望洁净的心中，即使天外漆黑的阴霾出现也无法侵蚀它们的身形。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | 灵光颂的内容主要是描述了灵光的力量和作用，以及它对人类和自然的祝福和保护。其中提到了灵光洗净亡者的罪孽，帮助人们超度到无悲苦的净土；灵光也洁净高天的渊薮，使污泥退去，慑服于正法；灵光将指引人们上升的道路。灵光还滋养着花海的善灵，保护着草木之神的末裔和百战劫余的勇士，使它们身形不受天外漆黑阴霾的侵蚀。                                                             | 灵光颂的内容主要是对灵光的歌颂和祈求。诗中描述了得助于万种灵光的人物已经超度尘土，洗净罪孽，往生净土。同时，诗中还描绘了在黑暗中逆流而上的形象，以及他们对于善灵开辟道路，智慧慈光普照四方的作用。总的来说，灵光颂的内容充满积极向上、追求真理和智慧的意涵。 |

For more compare results, you can take a look at. [compare_resuilt](compare_result.md)


### Note
I recommand you run the demo on GPU (10GB gpu memory is enough, all examples have tested on single GTX 1080Ti or GTX 3060) <br/><br/>

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
file in the gradio scripts.<br/> * For the ability of understanding the query and context, i recommand you use Mistral-7B in Huggingface Inference Api or Intel/neural-chat in ollama. <br/> * For the ability of answer quality in Chinese, i recommand you Qwen-7B in ollama or ChatGLM3-6B in chatglm.cpp. 

<br/><br/>

<!-- CONTACT -->
## Contact

<!--
Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com
-->
svjack - https://huggingface.co/svjack - svjackbt@gmail.com - ehangzhou@outlook.com

<!--
Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)
-->
Project Link:[https://github.com/svjack/Genshin-Impact-BookQA-LLM](https://github.com/svjack/Genshin-Impact-BookQA-LLM)


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [Genshin Impact](https://genshin.hoyoverse.com/en/)
* [Huggingface](https://huggingface.co)
* [Mistral-7B](https://mistral.ai/news/announcing-mistral-7b/)
* [LangChain](https://github.com/langchain-ai/langchain)
* [SetFit](https://github.com/huggingface/setfit)
* [HayStack](https://github.com/deepset-ai/haystack)
* [chatglm.cpp](https://github.com/li-plus/chatglm.cpp)
* [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
* [ollama](https://github.com/ollama/ollama)
* [svjack](https://huggingface.co/svjack)





