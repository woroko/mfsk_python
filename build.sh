p4a apk --requirements=python2,kivy==master,numpy,audiostream --private . --package=org.example.myapp --name "RXTX" --version 0.1 \
--bootstrap=sdl2 --java-build-tool gradle --debug --sdk-dir ~/Projects --ndk-dir ~/Projects/android-ndk-r17 --android-api 19 \
--arch=armeabi-v7a --ndk-version r17
