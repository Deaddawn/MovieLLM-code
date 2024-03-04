import openai
import os
import argparse
import json
import ast
from tqdm import tqdm



def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-eval-using-gpt-3")
    parser.add_argument("--output_dir", required=True, help="The path to save annotation json files.")
    parser.add_argument("--api_key", required=True, type=str, help="OpenAI API key")
    parser.add_argument("--api_base", default=None, type=str, help="OpenAI API base")
    parser.add_argument("--gt_dir", required=True, help="gt answer")
    parser.add_argument("--method_dir", required=True)
    parser.add_argument("--base_dir", required=True, help="baseline")
    args = parser.parse_args()
    return args




def main():
    args = parse_args()
    openai.api_key = args.api_key # Your API key here
    if args.api_base:
        openai.api_base = args.api_base # Your API base here
    gtp=args.gt_dir
    ourp=args.method_dir
    llamap=args.base_dir
    folder=os.listdir(gtp)
    folder.sort()
    for item in tqdm(folder):
        flag=0
        # print(item)
        resjsp=os.path.join(args.output_dir,item)
        if os.path.exists(resjsp):
            print('already exist, continue')
            continue
        gtjs=json.load(open(os.path.join(gtp,item)))
        ourjs=json.load(open(os.path.join(ourp,item)))
        llamajs=json.load(open(os.path.join(llamap,item)))
        resjs={
            "movie_title":gtjs['movie_title'],
            "QA_res":{
                "overview_qa": [],
                "plot_qa": [],
                "temporal_qa": []
            }
        }
        for k in gtjs['QA'].keys():
            for i in range(len(gtjs['QA'][k])):
                question=gtjs['QA'][k][i]['Question']
                gt_answer=gtjs['QA'][k][i]['Answer']
                our_answer=ourjs['QA'][k][i]['Answer']
                llama_answer=llamajs['QA'][k][i]['Answer']
                # print(question)
                try:
                    completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
               
                    {
                        "role": "user",
                        "content":
                            "You are an intelligent chatbot designed for comparing two video-based question-answer pairs to decide which one is better and score them, use score from 0 to 5, float is allowed based on the reference answer.\n\n"
                            "##Instructions:\n"
                            "1. consider the detailed information involved in the answer (more detailed, better)\n"
                            "2. consider the character relationship in the answer\n"
                            "3. consider the conclusion or ending involved in the answer\n"
                            "4. nonsense like repeated sentences are not allowed, should be considered as a very bad answer\n"
                            "5. if mark number is used in the answer, the order of the number should be right\n\n"
                            "Note that your answer should be like this: {'better one':'first one', 'score':{'first one':3.5,'second one':1.5}}, only choose between first and second answer,DO NOT PROVIDE ANY OTHER TEXT OR EXPLANATION, only provide the python dictionary string like above.\n\n"
                            f"Question: {question}\n\n"
                            f"Reference answer: {gt_answer}\n\n"
                            f"First answer: {our_answer}\n\n"
                            f"Second answer: {llama_answer}\n\n"
                            "Now, give me your answer."
                    }
                ]
            )
  
                    response_message = completion["choices"][0]["message"]["content"]
                    response_dict = ast.literal_eval(response_message)
                    resjs['QA_res'][k].append(response_dict)
        
                except Exception as e:
                    print('error process',item)
                    flag=1
        if flag==1:
            continue
        else:
            f=open(resjsp,'w')
            json.dump(resjs,f)
            f.close()
        
    if len(folder)!=len(os.listdir(args.output_dir)):
        print('Not all file complete, please re-run this script')
    else:
        print('all file complete')

if __name__ == "__main__":
    main()

