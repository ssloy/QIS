# QIS
Quasi-isometric stiffening C++ implementation

# Compile and run:
```sh
git clone --recurse-submodules https://github.com/ssloy/qis &&
cd qis &&
mkdir build &&
cd build &&
cmake .. &&
make -j &&
./stiffening2d ../hand.obj &&
./stiffening3d ../wrench-rest.mesh ../wrench-init.mesh ../wrench-lock.txt
```

