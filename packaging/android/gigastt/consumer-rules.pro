# Keep the UniFFI-generated JNA-bound bindings and JNA's native callback glue.
# UniFFI Kotlin loads the native library via JNA reflection, so the binding
# classes and JNA's Structure/Callback machinery must survive R8/ProGuard.
-keep class com.github.ekhodzitsky.gigastt.** { *; }
-keep class com.sun.jna.** { *; }
-keepclassmembers class * extends com.sun.jna.** { *; }
