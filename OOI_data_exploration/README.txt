Accessing OOI data using python and M2M
[A] https://github.com/oceanobservatories/ooi-data-explorations
[B] https://github.com/oceanobservatories/ooi-data-explorations/tree/master/python

## GETTING STARTED
# 
# A python subfolder with dataexploration scripts; instrument class descriptions; 
# and a workflow template are found here:
# https://github.com/oceanobservatories/ooi-data-explorations
# my code is below and for comparison, the OOI examples are in their readme (found at link above)

# Git workflow template for working with the OOI Data Explorations repository.

# Create your development directories (just a guide, use your own directories)
cd ~/Documents/LKH/OOI_data_exploration

# Fork the oceanobservatories/ooi-data-explorations repository to your account 
# and clone a copy of your fork to your development machine.
# git clone git@github.com:<your_account>/ooi-data-explorations.git
git clone git@github.com:katehewett/ooi-data-explorations.git
 
# The next steps must be completed in the local repository directory
cd ooi-data-explorations
 
# Add the upstream feed for the master repository
git remote add upstream git@github.com:oceanobservatories/ooi-data-explorations.git
git fetch upstream

# Set the local master to point instead to the upstream master branch
git branch master --set-upstream-to upstream/master

# Keep your master branch updated, tied to the upstream master, and
# keep your remote fork in sync with the official repository (do this
# regularly)
git pull --ff-only upstream master
git push origin master

# Create your feature branch based off of the most recent version of the master
# branch by starting a new branch via...
#    git checkout master
#    git pull
#    git push origin master
# ... and then:
git checkout -b <branch>

### --- All of the next steps assume you are working in your <branch> --- ###
# Do your work, making incremental commits as/if needed, and back up to your
# GitHub repository as/if needed.
while working == true
    git add <files>
    git commit -am "Commit Message"
    git push origin <branch>
end

# Before pushing your final changes to your repository, rebase your changes
# onto the latest code available from the upstream master.
git fetch upstream
git rebase -p upstream/master

# At this point you will need to deal with any conflicts, of which there should
# be none. Hopefully...

# Push the current working, rebased branch to your GitHub fork and then 
# make a pull request to merge your work into the main code branch. Once the
# pull request is generated, add a comment with the following text:
#
#    @<code_admin> Ready for review and merge
#
# This will alert the main code admin to process the pull request.
git push -f origin <branch>
 
# At this point you can switch back to your master branch. Once the pull
# request has been merged into the main code repository, you can delete
# your working branches both on your local machine and from your GitHub
# repository.
git checkout master
git pull
git push origin master
git branch -D <branch>
git branch -D origin/<branch>



I have git + python installed + access credentials set up. 
If you do too, then you can skip on down to "access credentials"
Login and password info can be found under your user profile.
In June 2023, if you go here: https://ooinet.oceanobservatories.org/ 
and if you're logged in, then on the map you'll see your login email in the upper right hand side of the map
click it / select user profile and grab your login (API username) and password (API token)
Note if you refresh your API token, then you need to also select submit and then close. 

My code looks like:

cd ~
touch .netrc
chmod 600 .netrc
cat <<EOT >> .netrc
machine ooinet.oceanobservatories.org
    login OOIAPI-PE271ZWB15VTF9
    password JBWM8ENQ62E
EOT

