
# README for getting started with the OOI Data Explorations repository.
  
  This covers initial steps to access OOI data using M2M & scripts from OOI data explorations, here focused on python
  Scripts and readme's found here: 
  (https://github.com/oceanobservatories/ooi-data-explorations)
  
  Note that directions under their readme to add the upstream feed for the master repository wasn't working for me.
  I kept getting the following error msg, after doing: 
      git fetch upstream
	  
      git@github.com: Permission denied (publickey).
      fatal: Could not read from remote repository.
 
      Please make sure you have the correct access rights
      and the repository exists.

Might need to figure out why. For the time being, I just cloned the repository to my laptop under "temp_repos", 
which is a folder that I save repositories I don't want on LKH 

On my laptop I did: 
cd /Users/katehewett/Documents/temp_repos 
git clone https://github.com/oceanobservatories/ooi-data-explorations.git

cd ooi-data-explorations/python

# ooi-data-explorations for python 

Since I have git+miniconda set up and access credentails at OOI, I skipped down to "access credentials"
If you have a OOI login, and you login here: (https://ooinet.oceanobservatories.org/) [June 2023] 
Then on the map you'll see your login email in the upper right hand side of a map.
Click it / select user profile and grab your login (API username) and password (API token)
Note if you refresh your API token, then you need to also select submit and then close. 

My code looks like:

	cd ~
	touch .netrc
	chmod 600 .netrc
	cat <<EOT >> .netrc
	machine ooinet.oceanobservatories.org
    	login my-API-username 
    	password my-API-token
	EOT




