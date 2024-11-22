
# from huggingface_hub import list_repo_refs
# out = list_repo_refs("allenai/OLMo-1B")
# branches = [b.name for b in out.branches]
# branches = branches[::10]
steps = stepss = ["14m", '31m', '70m', '160m', '410m', '1b', "1.4b", "2.8b"]

for i in steps:
    exec(open("extract_llm_activations.py").read(), {'arg': i})

# for step in steps:
#     pass
    # exec(open("extract_llm_activations.py").read(), {'arg': step})

for step in steps:
    exec(open("fit_average_subject.py").read(), {'arg': step})
