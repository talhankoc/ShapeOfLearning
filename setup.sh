#!/bin/sh

#Updates all packages, sets up git, clones repo, installs julia
GITEMAIL="kss45@duke.edu"
GITUSER="Kunaal Sharma"
GITREPO="https://github.com/talhankoc/ShapeOfLearning.git"
GITPATH="/home/ec2-user/"

cd ${GITPATH}
echo "y" | sudo yum update
echo "y" | sudo yum install git-core
echo "y" | sudo yum install python-devel
echo "y" | sudo yum install gcc
sudo pip install scipy
sudo pip install numpy
sudo pip install matplotlib
git config --global user.name ${GITUSER}
git config --global user.email ${GITEMAIL}
git clone ${GITREPO}
sudo tar -xvzf "ShapeOfLearning/Binaries/julia-1.0.1-linux-x86_64.tar.gz"
sudo PATH=${PATH}:${GITPATH}/julia-1.0.1/bin/
export $PATH