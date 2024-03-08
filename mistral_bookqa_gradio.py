from gradio_client import Client
client = Client("https://svjack-entity-property-extractor-zh.hf.space")

import pandas as pd
import numpy as np
import os
import re

from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain import chains
from rapidfuzz import fuzz

from huggingface_hub import snapshot_download

if not os.path.exists("genshin_book_chunks_with_qa_sp"):
    path = snapshot_download(
        repo_id="svjack/genshin_book_chunks_with_qa_sp",
        repo_type="dataset",
        local_dir="genshin_book_chunks_with_qa_sp",
        local_dir_use_symlinks = False
    )

if not os.path.exists("bge_small_book_chunks_prebuld"):
    path = snapshot_download(
        repo_id="svjack/bge_small_book_chunks_prebuld",
        repo_type="dataset",
        local_dir="bge_small_book_chunks_prebuld",
        local_dir_use_symlinks = False
    )

if not os.path.exists("mistral-7b"):
    path = snapshot_download(
        repo_id="svjack/mistral-7b",
        repo_type="model",
        local_dir="mistral-7b",
        local_dir_use_symlinks = False
    )

'''
query = "警察是如何破获邪恶计划的？" ## 警 执律 盗
k = 10
uniform_recall_docs_to_pairwise_cos(
    query,
    docsearch_bge_loaded.similarity_search_with_score(query, k = k, ),
    bge_book_embeddings
)
'''
def uniform_recall_docs_to_pairwise_cos(query ,doc_list, embeddings):
    assert type(doc_list) == type([])
    from langchain.evaluation import load_evaluator
    from langchain.evaluation import EmbeddingDistance
    hf_evaluator = load_evaluator("pairwise_embedding_distance", embeddings=embeddings,
                              distance_metric = EmbeddingDistance.COSINE)
    return sorted(pd.Series(doc_list).map(lambda x: x[0].page_content).map(lambda x:
        (x ,hf_evaluator.evaluate_string_pairs(prediction=query, prediction_b=x)["score"])
    ).values.tolist(), key = lambda t2: t2[1])

'''
sort_by_kw("深渊使徒", book_df)["content_chunks_formatted"].head(5).values.tolist() ### 深渊
'''
def sort_by_kw(kw, book_df):
    req = book_df.copy()
    req["sim_score"] = req.apply(
    lambda x:
    max(map(lambda y: fuzz.ratio(y, kw) ,eval(x["person"]) + eval(x["locate"]) + eval(x["locate"]))) if \
    eval(x["person"]) + eval(x["locate"]) + eval(x["locate"]) else 0
    , axis = 1
)
    req = req.sort_values(by = "sim_score", ascending = False)
    return req

def recall_chuncks(query, docsearch, embedding, book_df,
    sparse_threshold = 30,
    dense_top_k = 10,
    rerank_by = "emb",
):
    sparse_output = sort_by_kw(query, book_df)[["content_chunks_formatted", "sim_score"]]
    sparse_output_list = sparse_output[
        sparse_output["sim_score"] >= sparse_threshold
    ]["content_chunks_formatted"].values.tolist()
    dense_output = uniform_recall_docs_to_pairwise_cos(
        query,
        docsearch.similarity_search_with_score(query, k = dense_top_k,),
        embedding
    )
    for chunck, score in dense_output:
        if chunck not in sparse_output_list:
            sparse_output_list.append(chunck)
    if rerank_by == "emb":
        from langchain.evaluation import load_evaluator
        from langchain.evaluation import EmbeddingDistance
        hf_evaluator = load_evaluator("pairwise_embedding_distance", embeddings=embedding,
                                  distance_metric = EmbeddingDistance.COSINE)
        return pd.Series(sorted(pd.Series(sparse_output_list).map(lambda x:
            (x ,hf_evaluator.evaluate_string_pairs(prediction=query, prediction_b=x)["score"])
        ).values.tolist(), key = lambda t2: t2[1])).map(lambda x: x[0]).values.tolist()
    else:
        sparse_output_list = sorted(sparse_output_list, key = lambda x: fuzz.ratio(x, query), reverse = True)
    return sparse_output_list

def reduce_list_by_order(text_list, as_text = False):
    if not text_list:
        return
    df = pd.DataFrame(
        list(map(lambda x: (x.split("\n")[0], x.split("\n")[1], "\n".join(x.split("\n")[2:])), text_list))
    ).groupby([0, 1])[2].apply(list).map(lambda x: sorted(x, key = len, reverse=True)).map(
        "\n\n".join
    ).reset_index()
    d = dict(df.apply(lambda x: ((x.iloc[0], x.iloc[1]), x.iloc[2]), axis = 1).values.tolist())
    #return df
    order_list = []
    for x in text_list:
        a, b = x.split("\n")[0], x.split("\n")[1]
        if not order_list:
            order_list = [[a, [b]]]
        elif a in list(map(lambda t2: t2[0], order_list)):
            order_list[list(map(lambda t2: t2[0], order_list)).index(a)][1].append(b)
        elif a not in list(map(lambda t2: t2[0], order_list)):
            order_list.append([a, [b]])
    df = pd.DataFrame(pd.DataFrame(order_list).explode(1).dropna().apply(
        lambda x: (x.iloc[0], x.iloc[1], d[(x.iloc[0], x.iloc[1])]), axis = 1
    ).values.tolist()).drop_duplicates()
    if as_text:
        return "\n\n".join(
            df.apply(lambda x: "{}\n{}\n{}".format(x.iloc[0], x.iloc[1], x.iloc[2]), axis = 1).values.tolist()
        )
    return df

def build_gpt_prompt(query, docsearch, embedding, book_df, max_context_length = 4090):
    l = recall_chuncks(query, docsearch, embedding, book_df)
    context = reduce_list_by_order(l, as_text = True)
    context_l = []
    for ele in context.split("\n"):
        if sum(map(len, context_l)) >= max_context_length:
            break
        context_l.append(ele)
    context = "\n".join(context_l).strip()
    template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。

{context}

问题: {question}
有用的回答:"""
    return template.format(
        **{
            "context": context,
            "question": query
        }
    )

def collect_prompt_to_hist_list(prompt, add_assistant = False):
    l = pd.Series(prompt.split("\n\n")).map(lambda x: x.strip()).values.tolist()
    ll = []
    for ele in l:
        if not ll:
            ll.append(ele)
        else:
            if ele.startswith("文章标题：") or ele.startswith("问题:"):
                ll.append(ele)
            else:
                ll[-1] += ("\n\n" + ele)
    if add_assistant:
        ll_ = []
        for i in range(len(ll)):
            if i == 0:
                ll_.append((ll[i], "好的。"))
            elif i < len(ll) - 1:
                ll_.append((ll[i], "我读懂了。"))
            else:
                ll_.append((ll[i], ""))
        return ll_
    else:
        return ll

def row_to_content_ask(r):
    question = r["question"]
    content_list = r["content_list"]
    assert type(content_list) == type([])
    content_prompt_list = pd.Series(content_list).map(
        lambda x:  '''
    {}\n从上面的相关的叙述中抽取包含"{}"中词汇的相关语段。
    '''.format(x, question).strip()
    ).values.tolist()
    return content_prompt_list

def entity_extractor_by_llm(query, llm, show_process = False, max_length = 512):
    import re
    hist = [['请从下面的句子中提取实体和属性。不需要进行进一步解释。', '好的。'],
     ['宁波在哪个省份？', '实体：宁波 属性：省份'],
     ['中国的货币是什么？', '实体：中国 属性：货币'],
     ['百慕大三角在什么地方？', '实体：百慕大三角 属性：地方'],
     ['谁是最可爱的人？', "实体：人 属性：可爱"],
     ['黄河的拐点在哪里？', "实体：黄河 属性：拐点"],
     #["玉米的引进时间是什么时候？", ""]
     ]

    re_hist = pd.DataFrame(
    pd.Series(hist).map(
        lambda t2: (
            {
                "role": "user",
                "content": t2[0]
            },
            {
                "role": "assistant",
                "content": t2[1]
            },
        )
    ).values.tolist()).values.reshape([-1]).tolist()

    out = llm.create_chat_completion(
            messages = re_hist + [
                {
                    "role": "user",
                    #"content": prompt + "如果没有提到相关内容，请回答不知道。使用中文进行回答，不要包含任何英文。"
                    "content": query
                }
            ],
            stream=True
            )
    out_text = ""
    for chunk in out:
        delta = chunk['choices'][0]['delta']
        if "content" in delta:
            out_text += delta['content']
            #from IPython.display import clear_output
            #clear_output(wait=True)
            if show_process:
                print(out_text)
            if len(out_text) >= max_length:
                break
    e_list = re.findall(r"实体(.*?)属性", out_text.replace("\n", " "))
    if e_list:
        return re.findall(u"[\u4e00-\u9fa5]+" ,e_list[0])
    return None

def unzip_string(x, size = 2):
    if len(x) <= size:
        return [x]
    req = []
    for i in range(len(x) - size + 1):
        req.append(x[i: i + size])
    return req

def entity_extractor_by_adapter(x):
    import json
    result = client.predict(
    		x,	# str  in 'question' Textbox component
    		api_name="/predict"
    )
    with open(result, "r") as f:
        req = json.load(f)
    req_list = req.get("E-TAG", [])
    req_ = []
    for ele in req_list:
        for x in unzip_string(ele, 2):
            if x not in req_:
                req_.append(x)
    return req_

def query_content_ask_func(question, content_list,
        llm, setfit_model, show_process = False, max_length = 1024):
    ask_list = row_to_content_ask(
        {
            "question": question,
            "content_list": content_list
        }
    )
    #return ask_list
    req = []
    for prompt in ask_list:
        out = llm.create_chat_completion(
                messages = [
                    {
                        "role": "user",
                        "content": prompt + "如果没有提到相关内容，请回答不知道。使用中文进行回答，不要包含任何英文。"
                    }
                ],
                stream=True
                )
        out_text = ""
        for chunk in out:
            delta = chunk['choices'][0]['delta']
            if "content" in delta:
                out_text += delta['content']
                #from IPython.display import clear_output
                #clear_output(wait=True)
                if show_process:
                    print(out_text)
                if len(out_text) >= max_length:
                    break
        req.append(out_text)
    d = {
            "question": question,
            "content_list": content_list
        }
    assert len(req) == len(ask_list)
    d["question_content_relate_list"] = req
    d["relate_prob_list"] = setfit_model.predict_proba(
               req
        ).numpy()[:, 1].tolist()
    return d

def build_relate_ask_list(query, docsearch_bge_loaded, bge_book_embeddings, book_df,
                          llm, setfit_model, as_content_score_df = True,
                          show_process = False, add_relate_entities = False,
                          max_length = 1024):
    prompt = build_gpt_prompt(query, docsearch_bge_loaded, bge_book_embeddings, book_df)
    prompt_list = collect_prompt_to_hist_list(prompt)
    question = prompt_list[-1].split("\n")[0]
    content_list = prompt_list[1:-1]

    d = query_content_ask_func(question, content_list,
            llm, setfit_model, show_process = show_process)

    #entity_list = entity_extractor_by_llm(query, llm, show_process = show_process, max_length = max_length)
    entity_list = entity_extractor_by_adapter(query)
    if type(entity_list) != type([]):
        entity_list = []

    d["in_content_entity_list"] = list(map(lambda x:
        list(filter(lambda e: e in x, entity_list))
    , d["content_list"]))

    if add_relate_entities:
        relate_content_entity_list = [[]] * len(content_list)

        for entity in entity_list:
            entity_content_score_d = query_content_ask_func(entity, d["content_list"],
            llm, setfit_model, show_process = show_process)
            lookup_df = pd.DataFrame(
                list(zip(*[entity_content_score_d["content_list"],
                entity_content_score_d["relate_prob_list"]]))
            )
            for ii, (i, r) in enumerate(lookup_df.iterrows()):
                if r.iloc[1] >= 0.5 and entity not in relate_content_entity_list[ii]:
                    #relate_content_entity_list[ii].append(entity)
                    relate_content_entity_list[ii] = relate_content_entity_list[ii] + [entity]

        d["relate_content_entity_list"] = relate_content_entity_list

    if as_content_score_df:
        if add_relate_entities:
            df = pd.concat(
                [
                    pd.Series(d["content_list"]).map(lambda x: x.strip()),
                    pd.Series(d["in_content_entity_list"]),
                    pd.Series(d["relate_content_entity_list"]),
                    pd.Series(d["question_content_relate_list"]).map(lambda x: x.strip()),
                    pd.Series(d["relate_prob_list"])
                ], axis = 1
            )
            df.columns = ["content", "entities", "relate_entities", "relate_eval_str", "score"]
        else:
            df = pd.concat(
                [
                    pd.Series(d["content_list"]).map(lambda x: x.strip()),
                    pd.Series(d["in_content_entity_list"]),
                    #pd.Series(d["relate_content_entity_list"]),
                    pd.Series(d["question_content_relate_list"]).map(lambda x: x.strip()),
                    pd.Series(d["relate_prob_list"])
                ], axis = 1
            )
            df.columns = ["content", "entities", "relate_eval_str", "score"]
        req = []
        entities_num_list = df["entities"].map(len).drop_duplicates().dropna().sort_values(ascending = False).\
            values.tolist()
        for e_num in entities_num_list:
            req.append(
            df[
                df["entities"].map(lambda x: len(x) == e_num)
            ].sort_values(by = "score", ascending = False)
            )
        return pd.concat(req, axis = 0)
        #df = df.sort_values(by = "score", ascending = False)
        #return df
    return d

def mistral_predict(prompt, llm, show_process = True, max_length = 512):
    out = llm.create_chat_completion(
                messages = [] + [
                    {
                        "role": "user",
                        #"content": prompt + "如果没有提到相关内容，请回答不知道。使用中文进行回答，不要包含任何英文。"
                        "content": prompt
                    }
                ],
                stream=True
                )
    #from IPython.display import clear_output
    out_text = ""
    for chunk in out:
        delta = chunk['choices'][0]['delta']
        if "content" in delta:
            out_text += delta['content']
            if show_process:
                print(out_text)
            if len(out_text) >= max_length:
                break
            #clear_output(wait=True)
    #clear_output(wait=True)
    return out_text

def run_all(query, docsearch_bge_loaded, bge_book_embeddings, book_df,
                          llm, setfit_model, only_return_prompt = False):
    df = build_relate_ask_list(query, docsearch_bge_loaded, bge_book_embeddings, book_df,
                              llm, setfit_model, show_process=False)
    info_list = df[
        df.apply(
            lambda x: x["score"] >= 0.5 and bool(x["entities"]), axis = 1
        )
    ].values.tolist()
    if not info_list:
        return df, info_list, "没有相关内容，谢谢你的提问。"
    prompt = '''
问题: {}
根据下面的内容回答上面的问题，如果无法根据内容确定答案，请回答不知道。
{}
'''.format(query, "\n\n".join(pd.Series(info_list).map(lambda x: x[0]).values.tolist()))
    if only_return_prompt:
        return df, info_list, prompt
    out = mistral_predict(prompt + "\n使用中文进行回答，不要包含任何英文。", llm)
    return df, info_list, out

#book_df = pd.read_csv("genshin_book_chunks_with_qa_sp.csv")
book_df = pd.read_csv("genshin_book_chunks_with_qa_sp/genshin_book_chunks_with_qa_sp.csv")
book_df["content_chunks"].dropna().drop_duplicates().shape

book_df["content_chunks_formatted"] = book_df.apply(
        lambda x: "文章标题：{}\n子标题：{}\n内容：{}".format(x["title"], x["sub_title"], x["content_chunks"]),
        axis = 1
    )

texts = book_df["content_chunks_formatted"].dropna().drop_duplicates().values.tolist()

#embedding_path = "bge-small-book-qa/"
embedding_path = "svjack/bge-small-book-qa"
bge_book_embeddings = HuggingFaceEmbeddings(model_name=embedding_path)
docsearch_bge_loaded = FAISS.load_local("bge_small_book_chunks_prebuld/", bge_book_embeddings,
                                       allow_dangerous_deserialization = True
                                       )

from llama_cpp import Llama
#true_path = "mistral-7b-instruct-v0.2.Q4_0.gguf"
true_path = "mistral-7b/mistral-7b-instruct-v0.2.Q4_0.gguf"

#### 16g +
# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = Llama(
  model_path=true_path,  # Download the model file first
  n_ctx=8000,  # The max sequence length to use - note that longer sequence lengths require much more resources
  n_threads=8,            # The number of CPU threads to use, tailor to your system and the resulting performance
  n_gpu_layers=-1, # The number of layers to offload to GPU, if you have GPU acceleration available
chat_format="llama-2"
)

from setfit import SetFitModel
#setfit_model = SetFitModel.from_pretrained("setfit_info_cls")
setfit_model = SetFitModel.from_pretrained("svjack/setfit_info_cls")

import gradio as gr

with gr.Blocks() as demo:
    title = gr.HTML(
            """<h1 align="center"> <font size="+3"> Genshin Impact Book QA Mistral-7B Demo ⚡ </font> </h1>""",
            elem_id="title",
    )
    
    with gr.Column():
        with gr.Row():
            query = gr.Text(label = "输入问题：", lines = 1, interactive = True, scale = 5.0)
            run_button = gr.Button("得到答案")
        output = gr.Text(label = "回答：", lines = 5, interactive = True)
        recall_items = gr.JSON(label = "召回相关内容", interactive = False)

    with gr.Row():
        gr.Examples(
            [
             '丘丘人有哪些生活习惯？',
             '盐之魔神的下场是什么样的？',
             '岩王帝君是一个什么样的人？',
             '铳枪手的故事内容是什么样的？',
             '大蛇居住在哪里？',
             '珊瑚宫有哪些传说？',
             '灵光颂的内容是什么样的？',
             '枫丹有哪些故事？',
             '璃月有哪些故事？',
             '轻策庄有哪些故事？',
             '瑶光滩有哪些故事？',
             '稻妻有哪些故事？',
             '海祇岛有哪些故事？',
             '蒙德有哪些故事？',
             '璃月有哪些奇珍异宝？',
             '狸猫和天狗是什么关系？',
             '岩王帝君和归终是什么关系？',
            ],
            inputs = query,
            label = "被书目内容包含的问题"
        )
    with gr.Row():
        gr.Examples(
            [
             '爱丽丝女士是可莉的妈妈吗？',
             '摘星崖是什么样的？',
             '丘丘人使用的是什么文字？',
             '深渊使徒哪里来的？',
             '发条机关可以用来做什么？',
             '那先朱那做了什么？',
            ],
            inputs = query,
            label = "没有被书目明确提到的问题"
        )

    run_button.click(lambda x:
        run_all(x, docsearch_bge_loaded, bge_book_embeddings, book_df, llm,
                                setfit_model = setfit_model)[1:],
        query, [recall_items, output]
    )

demo.queue(max_size=4, concurrency_count=1).launch(debug=True, show_api=False, share = True)
