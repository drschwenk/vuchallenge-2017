competition/scoring.zip:
	cd scoring && zip ../competition/scoring.zip * && cd ..

competition/val_data.zip:
	cd dev_data && zip ../competition/dev_data.zip * && cd ..

competition/test_data.zip:
	cd test_data && zip ../competition/test_data.zip * && cd ..

competition.zip: competition/scoring.zip competition/dev_data.zip competition/test_data.zip
	cd competition && zip ../competition.zip * && cd ..

submission.zip:
	cd submission && zip ../submission.zip * && cd ..

clean:
	rm competition/*.zip
	rm competition/*.html
	rm competition/competition.yaml
	rm competition.zip

all:
	cd scoring && zip -r ../competition/scoring.zip * && cd ..
	cd val_data && zip ../competition/val_data.zip * && cd ..
	cd test_data && zip ../competition/test_data.zip * && cd ..
	cd html && cp * ../competition/
	cd competition && cp ../competition.yaml .
	cd competition && zip ../competition.zip * && cd ..
