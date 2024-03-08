<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">Genshin-Impact-BookQA-LLM</h3>

  <p align="center">
   		A Book Question Answer Project supported by LLM (build by LangChain ChatGLM Mistral OLlama)
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

This project is an attempt to build Chinese Q&A on the LLM support RAG system.

### Try Demo on the fly


|Name | HuggingFace Space link |
|---------|--------|
| Genshin Impact Book QA Haystack Demo 📈 | https://huggingface.co/spaces/svjack/genshin-impact-bookqa-haystack |

The Demo use Huggingface Inference Api that call [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) to perform Q&A tasks.
Because the base model is not fintuned in Chinese but have more better inference capabilities than most models. You can take this deploy version as a free preview version. 



