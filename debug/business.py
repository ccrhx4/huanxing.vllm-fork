from transformers import AutoTokenizer
import torch

device = "hpu"

if device == "hpu":
    import habana_frameworks.torch as ht

model_name = "Qwen/Qwen2-7B-Instruct"
message = "The following are multiple choice questions (with answers) about business. Think step by step and then finish your answer with \"the answer is (X)\" where X is the correct letter choice.\nQuestion:\nIn contrast to _______, _______ aim to reward favourable behaviour by companies. The success of such campaigns have been heightened through the use of ___________, which allow campaigns to facilitate the company in achieving _________ .\nOptions:\nA. Boycotts, Buyalls, Blockchain technology, Increased Sales\nB. Buycotts, Boycotts, Digital technology, Decreased Sales\nC. Boycotts, Buycotts, Digital technology, Decreased Sales\nD. Buycotts, Boycotts, Blockchain technology, Charitable donations\nE. Boycotts, Buyalls, Blockchain technology, Charitable donations\nF. Boycotts, Buycotts, Digital technology, Increased Sales\nG. Buycotts, Boycotts, Digital technology, Increased Sales\nH. Boycotts, Buycotts, Physical technology, Increased Sales\nI. Buycotts, Buyalls, Blockchain technology, Charitable donations\nJ. Boycotts, Buycotts, Blockchain technology, Decreased Sales\nAnswer: Let's think step by step. We refer to Wikipedia articles on business ethics for help. The sentence that best uses the possible options above is __n contrast to *boycotts*, *buycotts* aim to reward favourable behavior by companies. The success of such campaigns have been heightened through the use of *digital technology*, which allow campaigns to facilitate the company in achieving *increased sales*._ The answer is (F).\n\nQuestion:\n_______ is the direct attempt to formally or informally manage ethical issues or problems, through specific policies, practices and programmes.\nOptions:\nA. Operational management\nB. Corporate governance\nC. Environmental management\nD. Business ethics management\nE. Sustainability\nF. Stakeholder management\nG. Social marketing\nH. Human resource management\nI. N/A\nJ. N/A\nAnswer: Let's think step by step. We refer to Wikipedia articles on business ethics for help. The direct attempt manage ethical issues through specific policies, practices, and programs is business ethics management. The answer is (D).\n\nQuestion:\nHow can organisational structures that are characterised by democratic and inclusive styles of management be described?\nOptions:\nA. Flat\nB. Bureaucratic\nC. Autocratic\nD. Hierarchical\nE. Functional\nF. Decentralized\nG. Matrix\nH. Network\nI. Divisional\nJ. Centralized\nAnswer: Let's think step by step. We refer to Wikipedia articles on management for help. Flat organizational structures are characterized by democratic and inclusive styles of management, and have few (if any) levels of management between the workers and managers.  The answer is (A).\n\nQuestion:\nAlthough the content and quality can be as controlled as direct mail, response rates of this medium are lower because of the lack of a personal address mechanism. This media format is known as:\nOptions:\nA. Online banners.\nB. Television advertising.\nC. Email marketing.\nD. Care lines.\nE. Direct mail.\nF. Inserts.\nG. Door to door.\nH. Radio advertising.\nI. Billboards.\nJ. Social media advertising.\nAnswer: Let's think step by step. We refer to Wikipedia articles on marketing for help. Door to door marketing delivers non-addressed items within all buildings within a geographic area. While it can control the content and quality as well as direct mail marketing, its response rate is lower because of the lack of a personal address mechanism. The answer is (G).\n\nQuestion:\nIn an organization, the group of people tasked with buying decisions is referred to as the _______________.\nOptions:\nA. Procurement centre.\nB. Chief executive unit.\nC. Resources allocation group.\nD. Marketing department.\nE. Purchasing department.\nF. Supply chain management team.\nG. Outsourcing unit.\nH. Decision-making unit.\nI. Operations unit.\nJ. Financial management team.\nAnswer: Let's think step by step. We refer to Wikipedia articles on marketing for help. In an organization, the group of the people tasked with buying decision is referred to as the decision-making unit. The answer is (H).\n\nQuestion:\nManagers are entrusted to run the company in the best interest of ________. Specifically, they have a duty to act for the benefit of the company, as well as a duty of ________ and of _______.\nOptions:\nA. Shareholders, Diligence, Self-interest\nB. Shareholders, Self-interest, Care and Skill\nC. Stakeholders, Care and skill, Self-interest\nD. Stakeholders, Diligence, Care and Skill\nE. Customers, Care and Skill, Diligence\nF. Shareholders, Care and Skill, Diligence\nG. Shareholders, Self-interest, Diligence\nH. Employees, Care and Skill, Diligence\nI. Stakeholders, Self-interest, Diligence\nJ. Stakeholder, Care and Skill, Diligence\nAnswer: Let's think step by step."

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": message}
]

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

if device == "hpu":
    ht.core.mark_step()

inputs = tokenizer([message], return_tensors="pt").to(device)

generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=256, top_k=0, do_sample=False, repetition_penalty=1.0, top_p=1.0, temperature=1.0)
generated_ids = model.generate(**inputs, max_new_tokens=256, top_k=0, do_sample=False, repetition_penalty=1.0, top_p=1.0, temperature=1.0)

if device == "hpu":
    ht.core.mark_step()

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
