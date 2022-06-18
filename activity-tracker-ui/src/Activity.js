import {
	Box,
	Center,
	useColorModeValue,
	Heading,
	Text,
	Stack,
	Image,
} from '@chakra-ui/react';
import {useEffect, useState} from "react";
import {getMeasurement} from "./networking";

const IMAGE =
	'https://i.insider.com/5eea8a48f0f419386721f9e8?width=1136&format=jpeg';


const mapping = {
	WALKING: {
		title: 'Walking',
		img: 'https://i.insider.com/5eea8a48f0f419386721f9e8?width=1136&format=jpeg'
	},
	WALKING_UPSTAIRS: {
		title:'Walking upstairs',
		img: 'https://media.istockphoto.com/photos/man-wearing-suit-runs-up-the-stairs-picture-id684803840?k=20&m=684803840&s=612x612&w=0&h=DankgbEaLCguz3aAuU3BqBj92Gio9pqPC3-CusjHfJo='
	},
	WALKING_DOWNSTAIRS: {
		title:'Walking downstairs',
		img: 'https://media.istockphoto.com/photos/every-step-will-bring-you-closer-to-great-success-picture-id1132625382?k=20&m=1132625382&s=612x612&w=0&h=owbPmPLyiIuxVVJafZXKoJXe662zSn8J6CZA_T2fPk0='
	},
	SITTING: {
		title: 'Sitting',
		img: 'https://www.aimsindia.com/wp-content/uploads/2019/09/asian-blog-image1.jpg'
	},
	STANDING:{
		title: 'Standing',
		img: 'https://images.unsplash.com/photo-1527237545644-c3d2a74ede9f?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MXx8bWFuJTIwc3RhbmRpbmd8ZW58MHx8MHx8&w=1000&q=80'
	},
	LAYING:{
		title: 'Laying',
		img: 'https://merriam-webster.com/assets/mw/images/article/art-wap-landing-mp-lg/how-to-use-lay-and-lie-22@1x.jpg'
	}
}

export default function Activity() {
	const [res, setRes] = useState(null);

	useEffect(() => {
		getMeasurement().then(resp => setRes(resp))

		const interval = setInterval(() => {
			getMeasurement().then(resp => setRes(resp))
		}, 1000);
	}, [])

	const bg = useColorModeValue('white', 'gray.800')

	const active = res ? mapping[res.pred] : null;

	let color;
	if (res?.prob < 0.3) {
		color = 'red'
	}
	else if (res?.prob < 0.6) {
		color = 'orange'
	}
	else color='green'

	return (
		res !== null ?
		<Center py={12}>
			<Box
				role={'group'}
				p={6}
				maxW={'330px'}
				w={'full'}
				bg={bg}
				boxShadow={'2xl'}
				rounded={'lg'}
				pos={'relative'}
				zIndex={1}>
				<Box
					rounded={'lg'}
					mt={-12}
					pos={'relative'}
					height={'230px'}
					_after={{
						transition: 'all .3s ease',
						content: '""',
						w: 'full',
						h: 'full',
						pos: 'absolute',
						top: 5,
						left: 0,
						backgroundImage: `url(${active.img})`,
						filter: 'blur(15px)',
						zIndex: -1,
					}}
					_groupHover={{
						_after: {
							filter: 'blur(20px)',
						},
					}}>
					<Image
						rounded={'lg'}
						height={230}
						width={282}
						objectFit={'cover'}
						src={active.img}
					/>
				</Box>
				<Stack pt={10} align={'center'}>
					<Text color={'gray.500'} fontSize={'sm'} textTransform={'uppercase'}>
						Your current status is:
					</Text>
					<Heading fontSize={'2xl'} fontFamily={'body'} fontWeight={500}>
						{active.title}
					</Heading>
					<Stack direction={'row'} align={'center'}>
						<Text color={'gray.600'}>
							confidence:
						</Text>
						<Text fontWeight={500} fontSize={'lg'} color={color}>
							{(res.prob*100).toFixed(1)}%
						</Text>
					</Stack>
				</Stack>
			</Box>
		</Center> : <div/>
	);
}
