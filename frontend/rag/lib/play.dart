import 'package:flutter/material.dart';

/**
 * @author: Levi Muriuki
 */

class Play extends StatefulWidget {
  _PlayState createState() => new _PlayState();
}

class _PlayState extends State<Play> {
  bool _isInPlay;

  @override
  void initState() {
    // TODO: implement initState
    super.initState();
    _isInPlay = true;
  }

  @override
  Widget build(BuildContext context) {
    // TODO: implement build
    return Scaffold(
      appBar: AppBar(title: Text("Play"),),
      body: Builder(
        builder: (context) {
          return Column(
            children: <Widget>[
              Container(
                child: Align(
                  alignment: Alignment.topLeft,
                  child: FloatingActionButton(
                    child: Icon(
                      _isInPlay ? Icons.pause : Icons.play_arrow
                    ),
                    onPressed: () {
                      setState(() {
                        if (_isInPlay) {
                          _isInPlay = false;
                          Scaffold.of(context).showSnackBar(
                            SnackBar(
                              content: Text("Paused"),
                              duration: Duration(milliseconds: 500),
                            )
                          );
                        }
                        else {
                          _isInPlay = true;
                          Scaffold.of(context).showSnackBar(
                            SnackBar(
                              content: Text("Starting"),
                              duration: Duration(milliseconds: 500),
                            )
                          );
                        }
                      });
                    },
                  ),
                )
              ),
              Container(child: Placeholder(fallbackHeight: 350,),),
              Container(
                child: MaterialButton(
                  minWidth: 400.0,
                  height: 80,
                  color: Colors.lightGreen,
                  child: Text("Tap"),
                  onPressed: null,
                ),
              )
            ],
          );
        },
      )
    );
  }
}