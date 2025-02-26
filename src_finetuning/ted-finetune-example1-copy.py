# ===========================================================================
# Fine tuning 
# Updated: 26, Feb 2025
# Writer: Ted, Jung
# Description:  Just do only for fine-tuning
# ===========================================================================


import os
import nest_asyncio


from llama_index.finetuning import OpenAIFinetuneEngine


nest_asyncio.apply()

curr_dir = os.getcwd()

# FineTune (knowledge distillation)
def distil():
    finetune_engine = OpenAIFinetuneEngine(
        "gpt-4o-mini-2024-07-18",
        f"{curr_dir}/src_finetuning/data/correction_finetuning_events.jsonl"
    )

    finetune_engine.finetune()
    finetune_engine.get_current_job()


if __name__ =="__main__":
    # asyncio.run(ted_evaluate())
    distil()
    print("done")