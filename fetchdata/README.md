


# ETL

1. Download data (TNG, DS9, etc).
2. Split Episodes into scenes. 
4. Topic analysis on scenes.
3. Get main and recurrent characters (not chars who only have lines in one episode)
4. Get locations (generalize one-off locals)
5. For Each Scene: Write Characters, Topics, Location, Stemmed Scene.
6. word2vec handled by dl4j

e.g.

```
PICARD RIKER OBRIEN DATA GUEST4
0.029855 0.008702 0.000000 0.000000 0.021612 0.002922 0.000000 0.000000 0.009450 0.002050 0.111667 0.046659 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.018159 0.000000 0.000000 0.000000 0.000000 0.000000 0.064422 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.025212 0.022464 0.000000 0.000000 0.000000 0.016270 0.000000 0.000000 0.000000 0.023715 0.000000 0.000000 0.000000 0.000000 0.000000 0.018491 0.000000 0.000000 0.000000 0.004901 0.000000 0.003447 0.024047 0.000000 0.000000

[Battle Bridge]

(O'Brien is unaware the four were ever away) 
DATA: What is present course, conn? 
O'BRIEN: It's what it's been all along, sir. Direct heading to Farpoint
Station. 
DATA: Confirm. We are on that heading, sir. 
O'BRIEN: Know anything about Farpoint Station, sir? Sounds like a
fairly dull place. 
```

# Instructions to generate data

1. `python fetch_star_trek.py`
2. `python split_episodes_into_scenes.py` (populates `fetchdata/scenes/startrektng`)
3. `python generate-characters-file.py` (creates `fetchdata/scripts/tng-all-chars.txt`)
5. `python update-chars.py` (renames guest characters to things like `GUEST1`)
4. `python topic_analysis.py`

OR

```bash
python split_episodes_into_scenes.py
python generate-characters-file.py
python update-chars.py
python topic_analysis.py
```
# 1 Download Data

See `fetch_star_trek.py`

# 2 Split Episodes

See `split_episodes_into_scenes.py`


## TODO

Tag last scene on to last full scene if its too short. 