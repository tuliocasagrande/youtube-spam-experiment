Youtube Spam Experiment
======================
This is an experiment of filtering undesired comments from YouTube, using Scikit-learn.

----------
Database:
------------
The comments were extracted from 5 YouTube videos in October 2014, totaling 1,000 comments. The selected videos were among the most popular of YouTube during the study period and had between 30,000 and 5 million comments each.

A total of 200 comments from each video were labeled by hand -- (1) spam or (0) ham -- resulting in the following description:

\#| Video         | Spam | Ham
---| --------      | ---- | ---
1 | PlayStation4  | 20   | 180
2 | Avengers      | 40   | 160
3 | PSY           | 60   | 140
4 | KatyPerry     | 80   | 120
5 | PewDiePie     | 100  | 100

More information about the videos can be found [here][1].


[1]: https://github.com/tuliocasagrande/youtube-spam-experiment/blob/master/InfoVideos.md
