
# This README provides a workflow template for setting up an ooi-data-explorations repo 

This is a README that I wrote to help setup interactions with the ooi-data-explorations.  It also details adding SSH keys to your github, and basic how-tos on forks, git branch and remotes explained. 

>***Anticipating error(s)? Add an SSH key:*** GitHub deprecated password-based authentication for Git operations (circa 2021?), encouraging users to adopt more secure methods like PATs and SSH keys. If you don't have a SSH key set up, then you will likely run into an error when you try to fetch upstream. It's flagged in this README with a note. Try working through (Optional) Step X at the end of this README first, and then follow along with the README steps listed here.

<span style="color: red; font-weight: bold;">***KMH TODOs*** *[last update 27 March 2025]:*  
<input type="checkbox" checked> Email Chris about updates to ssh keys. Because can't follow install README as-is.  
<input type="checkbox" unchecked> Language: Master/Main: GitHub is replacing the term 'master' with 'main'. OOI uses master. Fix readme text / 2xcheck.  
<input type="checkbox" checked> Test pull request (and update README) after make a change that should be incorporated.    
<input type="checkbox" unchecked> Remove reminder notes at the end.  
<input type="checkbox" unchecked> Add trick to check your commits in your .bashrc.  

## STEP 1: Setup (or grab) your OOI login credentials 
**If you have an OOI login**, then you login here: [OOI site](https://ooinet.oceanobservatories.org/) [last accessed on Dec 2024]. 
Just above the map, upper right hand side, you'll see a login button (or your email if you're logged in).  
Once logged in, click your email / 'user profile' / and grab your login (API username) and password (API token).  
Note if you refresh your API token, then you need to also select submit and then close. 

**If you do not have an OOI login**, then go here: [setup ooinet](https://oceanobservatories.org/m2m/) and do:  
1. Create a user account on ooinet.oceanobservatories.org, or use the CILogon button with an academic or Google account.
2. Log in
3. Navigate to the drop down menu screen in the top-right corner menu
4. Click on the “User Profile” element of the drop down.
5. Copy and save the following data from the user profile: API Username and API Token.  

**Don't put your username/token in places that post to your public git**

## STEP 2: Fork ooi-data-explorations repo and clone to your machine

**On github, fork the oceanobservatories/ooi-data-explorations repository to your account:**  
Login to your GitHub account and navigate to the OOI data explorations repo: 
[OOI repo](https://github.com/oceanobservatories/ooi-data-explorations)  
In the top-right corner of the page, click Fork, and follow prompts.  

*You can read more on creating forks [here.](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo)*

**On your local machine:**  
**Create a directory that'll contain your cloned fork of ooi-data-explorations repo.**  
*This is just a guide, use your own directories. For me, this is on the same level that I have LO saved on my personal comuter.*  

**Clone a copy of the forked repo**  
*Note: You're cloning the forked version not the original version. On github.com, while on your forked repo page, find and click code/local/SSH and copy the url. This is the text that I have listed after git clone. Replace with your own version.* 
```  
mkdir -p ~/Documents/OBS_repos
cd ~/Documents/OBS_repos
git clone git@github.com:katehewett/ooi-data-explorations.git
```
*The clone of the repo is looking at the forked version.*  

In a standard setup, you generally have an origin and an upstream remote — the latter being the gatekeeper of the project or the source of truth to which you wish to contribute.  
In Terminal, do: 
```
cd ooi-data-explorations/
git remote -v 
```
This will verify that you have setup an origin, and you'll see that we still need to setup an upstream with the remote command. 

**Add upstream feed for the master repo**
```
# if not already in ooi-data-explorations folder on your machine, 
# then do:
# cd ooi-data-explorations/    
# else, do:

git remote add upstream git@github.com:oceanobservatories/ooi-data-explorations.git
git fetch upstream 
```
>***Error after trying to fetch?*** It's likely because you need to add a SSH key to your github settings. Try working through (Optional) Step X at the end of this README. Then try to `git fetch upstream` again. This should fix your problem :)

Verify that the remote is added correctly:
```
git remote -v
```
*You should now see both origin and upstream.* 

**Set the local master to point to the upstream master branch**
```
git branch master --set-upstream-to upstream/master
```
Keep your master branch updated, tied to the upstream master, and
keep your remote fork in sync with the official repository (do this regularly):
```
git pull --ff-only upstream master
git push origin master
```
*If the project has tags that have not merged to main you should also do: `git fetch upstream --tags` I did not do this here.*

## Step 3: Branches; Working with the repo and making contributions
Generally, you want to keep your local main branch as a close mirror of the upstream main and execute any work in feature branches, as they might later become pull requests. 

When you want to share some work with the upstream maintainers (here, the OOI crew) you branch off main, to create a feature branch. When satisfied, push it to your remote repository. And create a pull request. 

*It's important to make changes in a branch! Especially in a forked repo. It's basically a copy of the source files from the most recent version of the master (that we just updated above) -- but now you can make changes to them in a more organized way. This helps avoid conflicts and makes it easier on you.*

**To create a branch:**
```
#    We just did these commented-out steps... 
#    and everything should be up to date...
#    but starting a new branch would be like:
#    git checkout master
#    git pull
#    git push origin master
#    ... and then:
#    git checkout -b <yourbranch>
#    I did: 

git checkout -b ooi_kmh_dec2024
```

Right after you make a new feature branch, if you type: `git status`, it'll say something like:   

*On branch ooi_kmh_dec2024*  
*nothing to commit, working tree clean*  

Because we haven't made any changes. 

***To test pushing to OOI***, I made changes to the README that reflect Git's August 2021 change in password requirements. And saved. So now we have something to commit.
```
git status
git add README.md
# note the README file was red in terminal before I added it, and after adding it changes to green
git commit -am "Commit Message" 
git push origin ooi_kmh_dec2024
```

This sent my change up to github.  
On my github.com, I clicked 'Compare & pull request', and entered the following msg, as per OOI's request:  
`@<code_admin> Ready for review and merge`  
*This will alert the main code admin to process the pull request.*

And I wrote a quick description of my change underneath.  
And clicked 'Create Pull Request'

If you type: `git status`, it'll still say something like:   

*On branch ooi_kmh_dec2024*  
*nothing to commit, working tree clean*  

## STEP 4: Housekeeping, Deleting branches 
After you have made changes and submitted a pull request, you can switch back to your master branch. Once the pull request has been merged into the main code repository, you can delete
your working branches both on your local machine and from your GitHub
repository.    

In the example this README runs thru, my branch was named ooi_kmh_dec2024. I made edits, created a pull request, the request was merged, and I deleted the branches from my personal computer.

```
git checkout master
git pull
git push origin master
git branch -D <branch>
```
***---Have not completed [start]---***   
delete branch from GitHub repo::
Then in my GitHub repo ooi_kmh_dec2024 is stale, and I could delete there
or try:

git branch -D origin/<branch>

***---Have not completed [end]---***  

## (Optional) Step X:
Checking for ssh keys, adding ssh keys, and *hopefully* fixing the error code after trying to fetch upstream in step 2. Try:  

### 1. On your local machine, check for existing ssh keys  
Follow instructions on [how to check for ssh keys](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/checking-for-existing-ssh-keys).

### 2. If no ssh key for GitHub, generate a new ssh key and add to agent    
[Link](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) for github how-to Instructions.  

Plus an example, of what worked for me...  
In Terminal, on your local machine:  
- **Create a new SSH key, using your github email as a label:**
 

    ```
    ssh-keygen -t ed25519 -C "your_email@example.com"
    ``` 
    *"your_email@example.com" is your github email, e.g. "hewett.kate@gmail.com".*  
    And when prompted, you'll enter a passphrase 2x.

    *some threads mentione doing: `ssh-keygen -t rsa -b 4096 -C "your_email@example.com"` is better than ed25519, but I haven't tested it.  I defaulted to use github's rec.*

- **Add your SSH key to the ssh-agent:**
    ```
    eval "$(ssh-agent -s)"
    open ~/.ssh/config     # checks if file exists
    touch ~/.ssh/config    # if file doesn't exist  
    open ~/.ssh/config
    ```
    Enter text in config file, and use whatever you named your ssh key name above:

    ```
    Host github.com
        AddKeysToAgent yes
        UseKeychain yes
        IdentityFile ~/.ssh/id_ed25519
    ```
    Add your SSH private key to the ssh-agent and store your passphrase in the keycahin.  

    ```
    ssh-add --apple-use-keychain ~/.ssh/id_ed25519
    ```

### 3. Add new SSH key to your github  
To configure your account on GitHub.com to use your new (or existing) SSH key, you'll also need to add the key to your account. More instructions [here.](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)  
Plus an example, of what worked for me...  
- **Copy the SSH public key to your clipboard**   
If your SSH public key file has a different name than the example code, modify the filename to match your current setup. When copying your key, don't add any newlines or whitespace. In Terminal, do:
    ```
    pbcopy < ~/.ssh/id_ed25519.pub
    ```
    *This copies the contents of the id_ed25519.pub file to your clipboard*

- **Add SSH key to your Github profile**  
-- Login to your github.com profile, and in the upper-right corner the page, click your profile photo, then click Settings.  
-- In the "Access" section of the sidebar, click  SSH and GPG keys.  
Click New SSH key.  
-- In the "Title" field, add a descriptive label for the new key. For example, if you're using a personal laptop, you might call this key "Personal laptop".  
-- Select the type of key [Authentication Key].  
-- In the "Key" field, paste your public key.  
-- Click Add SSH key.

### 4. Test SSH Key 
You should now be able to execute git fetch upstream command 
Test SSH key by trying to fetch upstream again in Step 2: Setup directory on local machine and fork the OOI repo.

<span style="color: red; font-weight: bold;"> -----  
### <span style="color: red; font-weight: bold;"> Kate's reminder Notes and helpful links (remove when finalize):  
1. The git fetch command fetches all the changes from the repository. It's a safe command that does not modify your working directory or local branches.  
2. Use `git status -s` and `git log` or `git diff` to review changes before merging them into your local branches.  
3. To integrate the fetched changes, use git merge or git rebase. 

Links: 

[GitHub forks, git branch and remotes explained](https://www.google.com/search?q=fork+and+upstream+with+SSH+github&sca_esv=34e05643fe6f2863&udm=7&biw=1311&bih=656&sxsrf=ADLYWIKXaUQfpmIgev1isz6271gTQSGkIw%3A1733433579764&ei=6xhSZ5WnLtXa0PEP88yf6AQ&ved=0ahUKEwjVyI2mx5GKAxVVLTQIHXPmB00Q4dUDCA8&uact=5&oq=fork+and+upstream+with+SSH+github&gs_lp=EhZnd3Mtd2l6LW1vZGVsZXNzLXZpZGVvIiFmb3JrIGFuZCB1cHN0cmVhbSB3aXRoIFNTSCBnaXRodWIyCBAhGKABGMMEMgUQIRirAkj-EFC1B1i8D3ABeACQAQCYAWegAf8FqgEDOC4xuAEDyAEA-AEBmAIGoAKrA8ICBBAjGCfCAggQABiABBiiBMICCBAAGKIEGIkFwgIHECMYsAIYJ8ICChAhGKABGMMEGArCAgQQIRgKmAMAiAYBkgcBNqAHmSI&sclient=gws-wiz-modeless-video#fpstate=ive&vld=cid:18879ac7,vid:FnxFwyzm4Z4,st:0)

[Git Forks and Upstreams: How-to and a cool tip](https://www.atlassian.com/git/tutorials/git-forks-and-upstreams)