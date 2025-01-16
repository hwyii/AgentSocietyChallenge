## Local test result 
### First 100 tasks for yelp 
1. baseline result

Baseline result.

![baseline](yelp_100_baseline.png)

2. version 1.0

For yelp: we generate user style and add it into prompt for generating reviews and stars.
![user style](yelp_100_user.png)

3. version 2.0 

Based on version 1.0, we add reflection process to the class ReasoningBaseline.
![version 2.0](yelp_100_user_reflection.png)


4. version 3.0

For Amazon: we analysis item information and add it into prompt.
![item info amazon](amazon_100_item.png)

For goodreads: we analysis item information and add it into prompt.
![item info goodreads](goodreads_100_item.png)

For yelp: we analysis item information and add it into prompt.
![item info yelp](yelp_100_item.png)

4. version 4.0

For Amazon: Based on version 3.0, we add reflection to the reasoningbasline
![item info reflection amazon](amazon_100_item_reflection.png)

For goodreads: Based on version 3.0, we add reflection to the reasoningbasline
![item info reflection goodreads](goodreads_100_item_reflection.png)

For yelp: Based on version 3.0, we add reflection to the reasoningbasline, trying to debug
![item info reflection yelp](yelp_100_item_reflection.png)

Compared with version 3.0, no obvious improvement.
