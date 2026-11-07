* using python trame and quart, cookup a simple webapp to view image and mask overlay
sample url provided below, this hints when image and mask arg is updated, application needs to relod and query for the updated image and mask path, for now use a place holder for the path. the server will have direct access to these paths where the file extesion are nii.gz.
"localhost:80/?image=case0,mask=case0_lung"
+ use docker
+ test data is here /mnt/hd2/data/Totalsegmentator_dataset/s1345 
* trame and quart repos are  provided below
https://github.com/pallets/quart
https://github.com/kitware/trame

+ don't use a database, use just a csv file for now to lookup file paths
+ be as minalistic as possible, follow guidelines from https://github.com/dietrichgebert/ponytail , dont read the whole ponytail repo just use this https://github.com/DietrichGebert/ponytail/blob/main/.grok-plugin/marketplace.json
