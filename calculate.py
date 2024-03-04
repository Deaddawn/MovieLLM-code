import os
import json
import argparse



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="compare res")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    overview_total=100
    plot_total=200
    temporal_total=100
    our_overview=0
    our_plot=0
    our_temporal=0

    overview_score1=0
    plot_score1=0
    temporal_score1=0

    overview_score2=0
    plot_score2=0
    temporal_score2=0


    path=args.path
    folder=os.listdir(path)
    folder.sort()
    for item in folder:
        curjs=json.load(open(os.path.join(path,item)))
        if 'first' in curjs['QA_res']['overview_qa'][0]['better one'].lower():
            our_overview+=1
        if 'first' in curjs['QA_res']['plot_qa'][0]['better one'].lower():
            our_plot+=1
        if 'first' in curjs['QA_res']['plot_qa'][1]['better one'].lower():
            our_plot+=1
        if 'first' in curjs['QA_res']['temporal_qa'][0]['better one'].lower():
            our_temporal+=1
        overview_score1+=curjs['QA_res']['overview_qa'][0]['score']['first one']
        overview_score2+=curjs['QA_res']['overview_qa'][0]['score']['second one']
        plot_score1+=curjs['QA_res']['plot_qa'][0]['score']['first one']
        plot_score1+=curjs['QA_res']['plot_qa'][1]['score']['first one']
        plot_score2+=curjs['QA_res']['plot_qa'][0]['score']['second one']
        plot_score2+=curjs['QA_res']['plot_qa'][1]['score']['second one']
        temporal_score1+=curjs['QA_res']['temporal_qa'][0]['score']['first one']
        temporal_score2+=curjs['QA_res']['temporal_qa'][0]['score']['second one']

    print('overview ratio for first',our_overview/overview_total)
    print('plot ratio for first',our_plot/plot_total)
    print('temporal ratio for first',our_temporal/temporal_total)


    print('overview score','ours',overview_score1/overview_total,'llamavid',overview_score2/overview_total)
    print('plot score','ours',plot_score1/plot_total,'llamavid',plot_score2/plot_total)
    print('temporal score','ours',temporal_score1/temporal_total,'llamavid',temporal_score2/temporal_total)



if __name__ == "__main__":
    main()