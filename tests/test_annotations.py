from pathlib import Path


class TestGeoJSONannotations:
    def __init__(self):
        self.geojson_dict = self.json_strs()

    def write_temp_files(self, t_dir):
        path_dict = {}
        for poly_type in self.geojson_dict.keys():
            output_path = Path(t_dir) / "{}.json".format(poly_type)
            with open(output_path, mode="w") as f:
                f.writelines(self.geojson_dict[poly_type])

            path_dict[poly_type] = str(output_path)

        return path_dict

    def json_strs(self):
        json_dict = {
            "multi_class": """[
          {
            "type": "Feature",
            "id": "PathAnnotationObject",
            "geometry": {
              "type": "Polygon",
              "coordinates": [
                [
                  [10783, 11994],
                  [10400, 12761],
                  [10952, 12623],
                  [10783, 11994]
                ]
              ]
            },
            "properties": {
              "classification": {
                "name": "class1",
                "colorRGB": -14568209
              },
              "isLocked": false,
              "measurements": []
            }
          },
          {
            "type": "Feature",
            "id": "PathAnnotationObject",
            "geometry": {
              "type": "Polygon",
              "coordinates": [
                [
                  [20402, 9671],
                  [20119, 10160],
                  [20299, 10932],
                  [20788, 10392],
                  [20402, 9671]
                ]
              ]
            },
            "properties": {
              "classification": {
                "name": "class1",
                "colorRGB": -14568209
              },
              "isLocked": false,
              "measurements": []
            }
          },
          {
            "type": "Feature",
            "id": "PathAnnotationObject",
            "geometry": {
              "type": "Polygon",
              "coordinates": [
                [
                  [22049, 9916],
                  [21354, 10482],
                  [21689, 11356],
                  [22255, 10868],
                  [22267, 10855],
                  [22049, 9916]
                ]
              ]
            },
            "properties": {
              "classification": {
                "name": "class1",
                "colorRGB": -14568209
              },
              "isLocked": false,
              "measurements": []
            }
          },
          {
            "type": "Feature",
            "id": "PathAnnotationObject",
            "geometry": {
              "type": "Polygon",
              "coordinates": [
                [
                  [18604, 14745],
                  [18371, 14750],
                  [18403, 14981],
                  [18403, 14983],
                  [18638, 15193],
                  [18942, 14858],
                  [18604, 14745]
                ]
              ]
            },
            "properties": {
              "classification": {
                "name": "class2",
                "colorRGB": -14555795
              },
              "isLocked": false,
              "measurements": []
            }
          },
          {
            "type": "Feature",
            "id": "PathAnnotationObject",
            "geometry": {
              "type": "Polygon",
              "coordinates": [
                [
                  [19793, 13913],
                  [19594, 14083],
                  [19528, 14397],
                  [19685, 14475],
                  [19968, 14255],
                  [19793, 13913]
                ]
              ]
            },
            "properties": {
              "classification": {
                "name": "class2",
                "colorRGB": -14555795
              },
              "isLocked": false,
              "measurements": []
            }
          },
          {
            "type": "Feature",
            "id": "PathAnnotationObject",
            "geometry": {
              "type": "Polygon",
              "coordinates": [
                [
                  [9125, 12838],
                  [9018, 13160],
                  [9233, 13498],
                  [9386, 13483],
                  [9448, 12838],
                  [9125, 12838]
                ]
              ]
            },
            "properties": {
              "classification": {
                "name": "class2",
                "colorRGB": -14555795
              },
              "isLocked": false,
              "measurements": []
            }
          },
          {
            "type": "Feature",
            "id": "PathAnnotationObject",
            "geometry": {
              "type": "Polygon",
              "coordinates": [
                [
                  [12564, 10182],
                  [12380, 10827],
                  [12641, 10858],
                  [12810, 10443],
                  [12564, 10182]
                ]
              ]
            },
            "properties": {
              "classification": {
                "name": "class3",
                "colorRGB": -14603624
              },
              "isLocked": false,
              "measurements": []
            }
          },
          {
            "type": "Feature",
            "id": "PathAnnotationObject",
            "geometry": {
              "type": "Polygon",
              "coordinates": [
                [
                  [21379, 11984],
                  [21097, 12054],
                  [20954, 12394],
                  [21440, 12420],
                  [21443, 12420],
                  [21379, 11984]
                ]
              ]
            },
            "properties": {
              "classification": {
                "name": "class3",
                "colorRGB": -14603624
              },
              "isLocked": false,
              "measurements": []
            }
          },
          {
            "type": "Feature",
            "id": "PathAnnotationObject",
            "geometry": {
              "type": "Polygon",
              "coordinates": [
                [
                  [19568, 14632],
                  [19403, 14758],
                  [19322, 15041],
                  [19426, 15153],
                  [19725, 14941],
                  [19568, 14632]
                ]
              ]
            },
            "properties": {
              "classification": {
                "name": "class3",
                "colorRGB": -14603624
              },
              "isLocked": false,
              "measurements": []
            }
          }
        ]""",
            "non_contig": """[
          {
            "type": "Feature",
            "id": "PathAnnotationObject",
            "geometry": {
              "type": "Polygon",
              "coordinates": [
                [
                  [11187, 10978],
                  [8009, 12191],
                  [8009, 12207],
                  [9943, 16672],
                  [9959, 16643],
                  [10619, 17426],
                  [12906, 16689],
                  [13674, 14525],
                  [11187, 10978]
                ]
              ]
            },
            "properties": {
              "classification": {
                "name": "non_contig_roi",
                "colorRGB": -16215175
              },
              "isLocked": false,
              "measurements": []
            }
          },
          {
            "type": "Feature",
            "id": "PathAnnotationObject",
            "geometry": {
              "type": "Polygon",
              "coordinates": [
                [
                  [9564, 12591],
                  [9488, 12788],
                  [9617, 13128],
                  [9806, 12954],
                  [9700, 12667],
                  [9700, 12659],
                  [9685, 12652],
                  [9564, 12591]
                ]
              ]
            },
            "properties": {
              "classification": {
                "name": "interior",
                "colorRGB": -956495
              },
              "isLocked": false,
              "measurements": []
            }
          },
          {
            "type": "Feature",
            "id": "PathAnnotationObject",
            "geometry": {
              "type": "Polygon",
              "coordinates": [
                [
                  [18401, 11093],
                  [18197, 11154],
                  [18203.8, 11373.39],
                  [18128, 11517],
                  [18265, 11661],
                  [18529, 11555],
                  [18560, 11328],
                  [18401, 11093]
                ]
              ]
            },
            "properties": {
              "classification": {
                "name": "interior",
                "colorRGB": -956495
              },
              "isLocked": false,
              "measurements": []
            }
          },
          {
            "type": "Feature",
            "id": "PathAnnotationObject",
            "geometry": {
              "type": "Polygon",
              "coordinates": [
                [
                  [20428, 7816],
                  [17143, 8031],
                  [16698, 10994],
                  [17036, 13419],
                  [18709, 14663],
                  [18740, 14663],
                  [21104, 13772],
                  [21887, 10656],
                  [21887, 10641],
                  [20444, 7816],
                  [20428, 7816]
                ]
              ]
            },
            "properties": {
              "classification": {
                "name": "non_contig_roi",
                "colorRGB": -16215175
              },
              "isLocked": false,
              "measurements": []
            }
          }
        ]""",
            "tissue_roi": """[
          {
            "type": "Feature",
            "id": "PathAnnotationObject",
            "geometry": {
              "type": "Polygon",
              "coordinates": [
                [
                  [9299, 10685],
                  [9007, 10992],
                  [8976, 10992],
                  [8976, 11007],
                  [9207, 11422],
                  [9237, 11422],
                  [9759, 11514],
                  [9882, 11422],
                  [9698, 10900],
                  [9667, 10900],
                  [9299, 10685]
                ]
              ]
            },
            "properties": {
              "classification": {
                "name": "interior",
                "colorRGB": -956495
              },
              "isLocked": false,
              "measurements": []
            }
          },
          {
            "type": "Feature",
            "id": "PathAnnotationObject",
            "geometry": {
              "type": "Polygon",
              "coordinates": [
                [
                  [13904, 2395],
                  [13889, 2410],
                  [13858, 2457],
                  [11340, 4606],
                  [10051, 6862],
                  [3511, 12911],
                  [3097, 13724],
                  [3097, 13740],
                  [2805, 14953],
                  [2805, 14968],
                  [2805, 15045],
                  [4248, 15628],
                  [4310, 15628],
                  [4878, 15828],
                  [4908, 15843],
                  [4908, 15889],
                  [4878, 15966],
                  [4571, 17823],
                  [4586, 17823],
                  [4601, 17869],
                  [4632, 17915],
                  [6228, 19343],
                  [6259, 19343],
                  [6275, 19358],
                  [8485, 19819],
                  [8516, 19834],
                  [10327, 21262],
                  [10312, 21277],
                  [10312, 21308],
                  [10312, 21339],
                  [11325, 23549],
                  [11387, 23549],
                  [14073, 24378],
                  [14088, 24394],
                  [14104, 24394],
                  [17803, 24225],
                  [18049, 22889],
                  [18049, 22874],
                  [18064, 22874],
                  [18632, 21201],
                  [18632, 21185],
                  [18648, 21155],
                  [18709, 21139],
                  [20091, 20970],
                  [20106, 20986],
                  [20152, 21032],
                  [20260, 21108],
                  [20382, 21201],
                  [24143, 23043],
                  [24236, 23043],
                  [28565, 16273],
                  [28764, 12312],
                  [28396, 10163],
                  [27260, 8766],
                  [27244, 8751],
                  [27137, 8689],
                  [15608, 2410],
                  [13904, 2395]
                ]
              ]
            },
            "properties": {
              "classification": {
                "name": "tissue_roi",
                "colorRGB": -12228925
              },
              "isLocked": false,
              "measurements": []
            }
          },
          {
            "type": "Feature",
            "id": "PathAnnotationObject",
            "geometry": {
              "type": "Polygon",
              "coordinates": [
                [
                  [12215, 9257],
                  [12277, 10040],
                  [12308, 10086],
                  [12737, 10777],
                  [13336, 10209],
                  [12983, 9411],
                  [12983, 9380],
                  [12968, 9380],
                  [12215, 9257]
                ]
              ]
            },
            "properties": {
              "classification": {
                "name": "interior",
                "colorRGB": -956495
              },
              "isLocked": false,
              "measurements": []
            }
          }
        ]""",
        }

        return json_dict
