"""
We will measure how top k blocks are concentrated across batches.

We batch GSM8K prompts, for batch in {1,2,4,8,16,32,64}
"""


from helper import load_model

