import 'dart:async';

import 'package:flutter/material.dart';
import 'package:rag/play.dart';

/**
 * @author: Levi Muriuki
 */

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'RAG',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: WelcomeScreen(),
    );
  }
}

class WelcomeScreen extends StatefulWidget {  
  @override
  _WelcomeScreenState createState() => new _WelcomeScreenState();
}

class _WelcomeScreenState extends State<WelcomeScreen> {
  startTimer() async {
    var _duration = new Duration(seconds: 3);
    return new Timer(_duration, navigationPage);
  }
  
  void navigationPage() {
    Navigator.pushReplacement(
      context, 
      MaterialPageRoute(
        builder: (context) => Play()
      ),
    );
  }

  @override
  void initState() {
    super.initState();
    setState(() {
      startTimer();
    });  
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(child: Image.asset("images/flower.jpg"),),
    );
  }
}
