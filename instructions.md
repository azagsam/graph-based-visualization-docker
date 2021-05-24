# Multilingual Text Exploration Tool

## How it works
This is a multilingual visualization tool. It takes raw text, tokenizes it into sentences, determines the importance 
of each sentence according to the graph-based model, and presents it in a figure that can be interactively explored. 
The importance of each sentence is indicated by how large the node is. The connections between sentences show how 
well they correlate.

## Usage
Graphs are constructed in two steps:

1) In the first row, you have three cards, from which you can configure data: demo datasets, upload your data, and 
   reload graph. The "Demo datasets" card contain preloaded experiments that show many ways to construct the graph. 
   After you run an experiment, it is saved into the memory, and can be reloaded fast with the "Reload graph" card. 
   "Upload you data" enables you to import your own data. Before you make an upload, you need to specify the 
   language of your text. You may want to cluster your data, or you have a csv file that contains pre-defined classes 
   (in that case, a csv file must contain pre-tokenized sentences, and two columns with names "class" and "sentence"). 
   
   
2) In the second row, the graph figure is presented on the left. On the right, the 'Adjust graph' card contains 
   options to update the graph. If you enter a keyword, all nodes that contain this keyword, will become the largest 
   in the graph. You can limit the number of sentences if you have a large set. For purely visualization purposes, 
   you can scale nodes size; higher number will emphasize larger nodes and make smaller ones smaller. Edges 
   threshold is a single cut-off value that either shows or hides connections between sentences; a very high value 
   will connect only sentences that highly correspond in their meaning (in extreme cases only duplicates). You may 
   also want to contextualize your data. 
   



