### Here is the steps of my investigation

1.Use shell script to get number of samples for each class.
```
for i in $(find . -maxdepth 1 -type d) ; do      hchan@hchans-MacBook-Pro
    echo -n $i": " ;
    (find $i -type f | wc -l) ;
done
.:    64733
./right:     2367
./eight:     2352
./cat:     1733
./tree:     1733
./bed:     1713
./happy:     1742
./go:     2372
./dog:     1746
./no:     2375
./wow:     1745
./nine:     2364
./left:     2353
./stop:     2380
./three:     2356
./_background_noise_:        7
./sheila:     1734
./one:     2370
./bird:     1731
./zero:     2376
./seven:     2377
./up:     2375
./marvin:     1746
./two:     2373
./house:     1750
./down:     2359
./six:     2369
./yes:     2377
./on:     2367
./five:     2357
./off:     2357
./four:     2372

```

2. No silence samples found,how to get the silence training data?
    + Go into details of the data discussion. There are some extra info I missed before:
        > The folder _background_noise_ contains longer clips of "silence" that you can break up and use as training input.
        
        > The first part of filename is the unique id of the pronouncer, the last part is the repeated times.
        
    + Get the solution from tf tutorial, and found that the TF tutorial is exactly the same with this competition
        ``` 
        parser.add_argument(
        '--wanted_words',
        type=str,
        default='yes,no,up,down,left,right,on,off,stop,go',
        help='Words to use (others will be added to an unknown label)',)
        ```
        
    - For now, I decide to use the background noise as silence and some other labels as unknown.
    
3.  Mimic the TF tutorial, rewrite it into a simpler version.
