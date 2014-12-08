Youtube Spam Experiment
======================
This is an experiment of filtering undesired comments from YouTube, using Scikit-learn.

----------
Database:
------------
The comments were extracted from 3 YouTube videos in November 2014, totaling 900 comments. The selected videos were among the most popular of YouTube during the study period and had between 90,000 and 5 million comments each.

A total of 300 comments from each video were labeled by hand -- (1) spam or (0) ham -- resulting in the following description:

\# | Video     | Spam | Ham
---| --------- | ---- | ---
1  | KatyPerry | 150  | 150
2  | PewDiePie | 150  | 150
3  | PSY       | 150  | 150

More information about the videos can be found [here][1].

From each original dataset was created 5 subsets, with 200 comments:

\# | Spam | Ham | Description
---| ---- | --- | -----------
1  | 50   | 150 | highly unbalanced (more ham)
2  | 75   | 125 | slightly unbalanced (more ham)
3  | 100  | 100 | balanced
4  | 125  | 75  | slightly unbalanced (more spam)
5  | 150  | 50  | highly unbalanced (more spam)


[1]: https://github.com/tuliocasagrande/youtube-spam-experiment/blob/master/InfoVideos.md
