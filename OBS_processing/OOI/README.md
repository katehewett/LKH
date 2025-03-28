
# This README provides a workflow template for accessing OOI data using python

This README assumes that you:   
    -are working with the LO system and run through Parker's 4 installation steps (see LO README). 
    -have OOI login credentials  
    -have forked the ooi data explorations repo and cloned it to your machine  

>***If this is your first time accessing OOI data / using the OOI data explorations***, NEWBIE_README provides instructions to help with setting ooi login credentials, and setting up the ooi data explorations repo on your machine/github.  

<span style="color: red; font-weight: bold;">***KMH TODOs***:*  
<input type="checkbox" unchecked> <span style="color: red; font-weight: bold;">Insert link  
<input type="checkbox" unchecked> <span style="color: red; font-weight: bold;">Look at ooi env and check for adjustments. Right now, it's just a combination where I added on loenv dependencies and the path to install lo_tools so we can use functions when interacting with ooi data 

## STEP 1: Create an (ooi) environment 
Create an environment that has all the modules required to interact with the ooi data exploration repo. One way to do this would be to:  
- copy the ooi [environment.yml](https://github.com/oceanobservatories/ooi-data-explorations/tree/master/python) to your machine. Rename it, for example, to ooi_environment.yml
- ***Or if you use LO tools,*** you can copy Kate's ooi environment which let's you use LO tools while working with ooi data explorations. <span style="color: red; font-weight: bold;">[INSERT LINK]
- edit the environment so that kh_ooi is on the first line (or whatever you want to call it)
- add or subtract any packages you like
- if you are using the lo_tools local package, change the path for it in the yml to be -e ../../../LO/lo_tools/ lo_tools (assuming that is the correct relative path) 
- then do: `conda env create -f ooi_environment.yml > env.log`

At any time you can do `conda info --envs` to find out what environments you have. And if you want to cleanly get rid of an environment, just make sure it is not active, then do `conda env remove -n myenv` or whatever name you are wanting to remove. This will not delete your yml file. 

## STEP 1: Grab your OOI login credentials 
Login here: [OOI site](https://ooinet.oceanobservatories.org/)  

- Just above the map, upper right hand side, you'll see a login button (or your email if you're logged in).  

- Once logged in, click your email / 'user profile' / and grab your login (API username) and password (API token).  

- Note if you refresh your API token, then you need to also select submit and then close.  

See NEWBIE_README.md if you don't have login creds.  

<span style="color: Blue; font-weight: bold;">**Don't put your username/token in places that post to your public git**

## STEP 2: 
