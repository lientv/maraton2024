#pragma GCC optimize("Ofast")
#pragma GCC target ("avx2")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,tune=native")
#pragma GCC optimize("unroll-loops")
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include "td/utils/base64.h"
#include "td/utils/buffer.h"
#include "td/utils/misc.h"
#include "vm/boc.h"

#define L(i,a,b) for (int i = (a), _=(b); i < _; ++i)
#define eb emplace_back
#define pb push_back

using U8=uint8_t;
using std::string;
using U16=uint16_t;
using U32=uint32_t;

char obuf[1<<24], tbuf[1<<24];
U8 first=0;
int clen=0, plen=0, dx=0;

struct Dat {
	char* const p;
	size_t l, r; 
	Dat(char* const s, size_t a, size_t b): p(s), l(a), r(b){}
};

int getc(Dat *d) {if(d->l >= d->r) return EOF; return d->p[d->l++];}
U32 getc(Dat *d, int bs) {
	U32 r = 0;
	while (bs--) r = (r<<8) + U8(getc(d));
	return r;
}
void putc(U8 c, Dat *d){d->p[d->r++] = c;}
void putc(U32 x, int bs, Dat *d) {
	L(i, 0, bs) {
		d->p[d->r+bs-1-i] = x & 255;
		x >>= 8;
	}
	d->r += bs;
}

#define MEM (1 << 20)
inline int min(int a, int b) { return a < b ? a : b; }
inline int max(int a, int b) { return a < b ? b : a; }
 
template<class T,int ALIGN=0> class Array {
private:
	int n, reserved;
	char *ptr;
	T *data;
	void create(int i);
public:
	explicit Array(int i = 0) { create(i); }
	~Array();
	T&operator[](int i) { return data[i]; }
	const T&operator[](int i) const { return data[i]; }
	int size() const { return n; }
private:
	Array(const Array&);
	Array&operator=(const Array&);
};
template<class T,int ALIGN> void Array<T,ALIGN>::create(int i) {
	n = reserved = i;
	if (i <= 0) {
		data = 0, ptr = 0;
		return;
	}
	const int sz = ALIGN + n * sizeof(T);
	ptr = (char*)calloc(sz, 1);
	if (!ptr) throw "Out of memory";
	data = (ALIGN ? (T*)(ptr + ALIGN - (((long)ptr) & (ALIGN - 1))):(T*)ptr);
}
template<class T,int ALIGN> Array<T,ALIGN>::~Array() {
	free(ptr);
}
 
static class Random {
	Array<U32> table;
	int i;
public:
	Random():table(64) {
		table[0] = 123456789;
		table[1] = 987654321;
		for (int j = 0; j < 62; j++) {
			table[j + 2] = table[j + 1] * 11 + table[j] * 23 / 16;
		}
		i = 0;
	}
	U32 operator()() {
		return ++i, table[i & 63] = table[(i - 24) & 63] ^ table[(i - 55) & 63];
	}
} rnd;
 
static int pos;
class Buf {
	Array<U8> b;
public:
	Buf(int i = 0):b(i) {}
	U8& operator[](int i) {
		return b[i & (b.size() - 1)];
	}
	int operator()(int i) const {
		return b[(pos - i) & (b.size() - 1)];
	}
	int size() const{
		return b.size();
	}
};
 
static int y = 0, c0 = 1, bpos = 0;
static U32 c4 = 0;
static Buf buf(MEM * 8);
 
static class Ilog {
	Array<U8> t;
public:
	int operator()(U16 x) const { return t[x]; }
	Ilog():t(65536) {
		U32 x = 14155776;
		for (int i = 2; i < 65536; ++i) {
			x += 774541002 / (i * 2 - 1);
			t[i] = x >> 24;
	}
}
} ilog;
 
static class Ptable {
	Array<int> t;
public:
	int operator()(U16 x) const { return t[x]; }
	Ptable():t(1024) {
		for (int i = 0; i < 1024; ++i) {
			t[i] = 16384 / (i + i + 3);
		}
	}
} pt;
 
static U32 hash(U32 a, U32 b, U32 c = 0xffffffff) {
	U32 h = a * 200002979u + b * 30005491u + c * 50004239u + 4114959990u;
	return h ^ h >> 9 ^ a >> 2 ^ b >> 3 ^ c >> 4 ^ 0x4000000;
}
 
static const U8 State_table[256][2] = {
	{1,2},{3,5},{4,6},{7,10},{8,12},{9,13},{11,14},{15,19},{16,23},{17,24},
	{18,25},{20,27},{21,28},{22,29},{26,30},{31,33},{32,35},{32,35},{32,35},
	{32,35},{34,37},{34,37},{34,37},{34,37},{34,37},{34,37},{36,39},{36,39},
	{36,39},{36,39},{38,40},{41,43},{42,45},{42,45},{44,47},{44,47},{46,49},
	{46,49},{48,51},{48,51},{50,52},{53,43},{54,57},{54,57},{56,59},{56,59},
	{58,61},{58,61},{60,63},{60,63},{62,65},{62,65},{50,66},{67,55},{68,57},
	{68,57},{70,73},{70,73},{72,75},{72,75},{74,77},{74,77},{76,79},{76,79},
	{62,81},{62,81},{64,82},{83,69},{84,71},{84,71},{86,73},{86,73},{44,59},
	{44,59},{58,61},{58,61},{60,49},{60,49},{76,89},{76,89},{78,91},{78,91},
	{80,92},{93,69},{94,87},{94,87},{96,45},{96,45},{48,99},{48,99},{88,101},
	{88,101},{80,102},{103,69},{104,87},{104,87},{106,57},{106,57},{62,109},
	{62,109},{88,111},{88,111},{80,112},{113,85},{114,87},{114,87},{116,57},
	{116,57},{62,119},{62,119},{88,121},{88,121},{90,122},{123,85},{124,97},
	{124,97},{126,57},{126,57},{62,129},{62,129},{98,131},{98,131},{90,132},
	{133,85},{134,97},{134,97},{136,57},{136,57},{62,139},{62,139},{98,141},
	{98,141},{90,142},{143,95},{144,97},{144,97},{68,57},{68,57},{62,81},{62,81},
	{98,147},{98,147},{100,148},{149,95},{150,107},{150,107},{108,151},{108,151},
	{100,152},{153,95},{154,107},{108,155},{100,156},{157,95},{158,107},{108,159},
	{100,160},{161,105},{162,107},{108,163},{110,164},{165,105},{166,117},
	{118,167},{110,168},{169,105},{170,117},{118,171},{110,172},{173,105},
	{174,117},{118,175},{110,176},{177,105},{178,117},{118,179},{110,180},
	{181,115},{182,117},{118,183},{120,184},{185,115},{186,127},{128,187},
	{120,188},{189,115},{190,127},{128,191},{120,192},{193,115},{194,127},
	{128,195},{120,196},{197,115},{198,127},{128,199},{120,200},{201,115},
	{202,127},{128,203},{120,204},{205,115},{206,127},{128,207},{120,208},
	{209,125},{210,127},{128,211},{130,212},{213,125},{214,137},{138,215},
	{130,216},{217,125},{218,137},{138,219},{130,220},{221,125},{222,137},
	{138,223},{130,224},{225,125},{226,137},{138,227},{130,228},{229,125},
	{230,137},{138,231},{130,232},{233,125},{234,137},{138,235},{130,236},
	{237,125},{238,137},{138,239},{130,240},{241,125},{242,137},{138,243},
	{130,244},{245,135},{246,137},{138,247},{140,248},{249,135},{250,69},{80,251},
	{140,252},{249,135},{250,69},{80,251},{140,252}};
							
static int squash(int d) {
	static const int t[33] = {
		1,2,3,6,10,16,27,45,73,120,194,310,488,747,1101,1546,2047,2549,2994,3348,
		3607,3785,3901,3975,4022,4050,4068,4079,4085,4089,4092,4093,4094};
	if (d > 2047) return 4095;
	if (d < -2047) return 0;
	int w = d & 127;
	d = (d >> 7) + 16;
	return (t[d] * (128 - w) + t[(d + 1)] * w + 64) >> 7;
}
 
class Stretch {
	Array<short> t;
public:
	int operator()(int x) const { return t[x]; }
	Stretch():t(4096) {
		int j = 0;
		for (int x = -2047; x <= 2047; ++x) {
			int i = squash(x);
			while (j <= i) t[j++] = x;
		}
		t[4095] = 2047;
	}
} stretch;
 
#include <emmintrin.h>
static int dot_product (const short* const t, const short* const w, int n) {
  __m128i sum = _mm_setzero_si128 ();
  while ((n -= 8) >= 0) {
    __m128i tmp = _mm_madd_epi16 (*(__m128i *) &t[n], *(__m128i *) &w[n]);
    tmp = _mm_srai_epi32 (tmp, 8);
    sum = _mm_add_epi32 (sum, tmp);
  }
  sum = _mm_add_epi32 (sum, _mm_srli_si128 (sum, 8));
  sum = _mm_add_epi32 (sum, _mm_srli_si128 (sum, 4));
  return _mm_cvtsi128_si32 (sum);
}
static void train (const short* const t, short* const w, int n, const int e) {
  if (e) {
    const __m128i one = _mm_set1_epi16 (1);
    const __m128i err = _mm_set1_epi16 (short(e));
    while ((n -= 8) >= 0) {
      __m128i tmp = _mm_adds_epi16 (*(__m128i *) &t[n], *(__m128i *) &t[n]);
      tmp = _mm_mulhi_epi16 (tmp, err);
      tmp = _mm_adds_epi16 (tmp, one);
      tmp = _mm_srai_epi16 (tmp, 1);
      tmp = _mm_adds_epi16 (tmp, *(__m128i *) &w[n]);
      *(__m128i *) &w[n] = tmp;
    }
  }
}

class Mixer {
	const int N, M, S;
	Array<short,16> tx, wx;
	Array<int> cxt, pr;
	int ncxt, base, nx;
	Mixer *mp;
public:
	Mixer(int n, int m, int s = 1, int w = 0):
	N((n + 7) & -8), M(m), S(s), tx(N), wx(N * M),
	cxt(S), pr(S), ncxt(0), base(0), nx(0), mp(0) {
		for (int i = 0; i < S; ++i) {
			pr[i] = 2048;
		}
		for (int j = 0; j < N * M; ++j) {
			wx[j] = w;
		}
		if (S > 1) mp = new Mixer(S, 1, 1);
	}
	void update() {
		for (int i = 0; i < ncxt; ++i) {
			int err = ((y << 12) - pr[i]) * 7;
			train(&tx[0], &wx[cxt[i] * N], nx, err);
		}
		nx = base = ncxt = 0;
	}
	void add(int x) { tx[nx++] = x; }
	void set(int cx, int range) {
		cxt[ncxt++] = base + cx;
		base += range;
	}
	int p() {
		while (nx & 7) tx[nx++] = 0;
		if (mp) {
			mp->update();
			for (int i = 0; i < ncxt; ++i) {
				pr[i] = squash(dot_product(&tx[0], &wx[cxt[i] * N], nx) >> 5);
				mp->add(stretch(pr[i]));
			}
			mp->set(0, 1);
			return mp->p();
		} else {
			return pr[0] = squash(dot_product(&tx[0], &wx[0], nx) >> 8);
		}
	}
	~Mixer() { delete mp; }
};
 
class APM {
	int index;
	const int N;
	Array<U16> t;
public:
	APM(int n):index(0), N(n), t(n * 33) {
		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < 33; ++j) {
				t[i * 33 + j] = i == 0 ? squash((j - 16) * 128) * 16 : t[j];
			}
		}
	}
	int p(int pr = 2048, int cxt = 0, int rate = 7) {
		pr = stretch(pr);
		int g = (y << 16) + (y << rate) - y - y;
		t[index] += (g - t[index]) >> rate;
		t[index + 1] += (g - t[index + 1]) >> rate;
		const int w = pr & 127;
		index = ((pr + 2048) >> 7) + cxt * 33;
		return(t[index] * (128 - w) + t[index + 1] * w) >> 11;
	}
};
 
class StateMap {
protected:
	const int N;
	int cxt;
	Array<U32> t;  
public:
	StateMap(int n = 256):N(n), cxt(0), t(n) {
		for (int i = 0; i < N; ++i) {
			t[i] = 1 << 31;
		}
	}
	int p(int cx) {
		U32 *p = &t[cxt], p0 = p[0];
		int n = p0 & 1023, pr = p0 >> 10;
		if (n < 1023) ++p0;
		else p0 = (p0 & 0xfffffc00) | 1023;
		p0 += (((y << 22) - pr) >> 3) * pt(n) & 0xfffffc00;
		p[0] = p0;
		return t[cxt = cx] >> 20;
	}
};
 
class ContextMap {
	const int C;
	class E {
		U16 chk[7];
		U8 last;
	public:
		U8 bh[7][7];
		U8* get(U16 chk);
	};
	Array<E, 64> t;
	Array<U8*> cp, cp0, runp;
	Array<U32> cxt;
	StateMap *sm;
	int cn;
	void update(U32 cx, int c);
public:
	ContextMap(int m, int c = 1);
	~ContextMap();
	void set(U32 cx, int next = -1);
	int mix(Mixer&m);
};
inline U8*ContextMap::E::get(U16 ch) {
	if (chk[last & 15] == ch) return &bh[last & 15][0];
	int b = 0xffff, bi = 0;
	for (int i = 0; i < 7; ++i) {
		if (chk[i] == ch) return last = last << 4 | i, (U8*) &bh[i][0];
		int pri = bh[i][0];
		if (pri < b && (last & 15) != i && last >> 4 != i) b = pri, bi = i;
	}
	return last = 0xf0 | bi, chk[bi] = ch, (U8*)memset(&bh[bi][0], 0, 7);
}
ContextMap::ContextMap(int m, int c):C(c), t(m>>6), cp(c), cp0(c), runp(c),
		cxt(c), cn(0) {
	sm = new StateMap[C];
	for (int i = 0; i < C; ++i) {
		cp0[i] = cp[i] = &t[0].bh[0][0];
		runp[i] = cp[i] + 3;
	}
}
ContextMap::~ContextMap() {
	delete[] sm;
}
inline void ContextMap::set(U32 cx, int next) {
	int i = cn++;
	i &= next;
	cx = cx * 987654323 + i;
	cx = cx << 16 | cx >> 16;
	cxt[i] = cx * 123456791 + i;
}
int ContextMap::mix(Mixer &m) {  
	int result = 0;
	for (int i = 0; i < cn; ++i) {
		if (cp[i]) {
			int ns = State_table[*cp[i]][y];
			if (ns >= 204 && rnd() << ((452 - ns) >> 3)) ns -= 4;
			*cp[i] = ns;
		}
		if (bpos > 1 && runp[i][0] == 0) {
			cp[i] = 0;
		} else {
			switch(bpos) {
				case 1: case 3: case 6: cp[i] = cp0[i] + 1 + (c0 & 1); break;
				case 4: case 7: cp[i] = cp0[i] + 3 + (c0 & 3); break;
				case 2: case 5: cp0[i] = cp[i] = t[(cxt[i] + c0) & (t.size() - 1)].get(cxt[i] >> 16); break;
				default: {
					cp0[i] = cp[i] = t[(cxt[i] + c0) & (t.size() - 1)].get(cxt[i] >> 16);
					if (cp0[i][3] == 2) {
						const int c = cp0[i][4] + 256;
						U8 *p = t[(cxt[i] + (c >> 6)) & (t.size() - 1)].get(cxt[i] >> 16);
						p[0] = 1 + ((c >> 5) & 1);
						p[1 + ((c >> 5) & 1)] = 1 + ((c >> 4) & 1);
						p[3 + ((c >> 4) & 3)] = 1 + ((c >> 3) & 1);
						p = t[(cxt[i] + (c >> 3)) & (t.size() - 1)].get(cxt[i] >> 16);
						p[0] = 1 + ((c >> 2) & 1);
						p[1 + ((c >> 2) & 1)] = 1 + ((c >> 1) & 1);
						p[3 + ((c >> 1) & 3)] = 1 + (c & 1);
						cp0[i][6] = 0;
					}
					int c1 = buf(1);
					if (runp[i][0] == 0) {
						runp[i][0] = 2, runp[i][1] = c1;
					} else if (runp[i][1] != c1) {
						runp[i][0] = 1, runp[i][1] = c1;
					} else if (runp[i][0] < 254) {
						runp[i][0] += 2;
					}
					runp[i] = cp0[i] + 3;
				} break;
			}
		}
		if ((runp[i][1] + 256) >> (8 - bpos) == c0) {
			int rc = runp[i][0];
			int b = (runp[i][1] >> (7 - bpos) & 1) * 2 - 1;
			int c = ilog(rc+1) << (2 + (~rc & 1));
			m.add(b * c);
		} else {
			m.add(0);
		}
		int p;
		if (cp[i]) {
			result += (*cp[i] > 0);
			p = sm[i].p(*cp[i]);
		} else {
			p = sm[i].p(0);
		}
		m.add(stretch(p));
	}
	if (bpos == 7) cn=0;
	return result;
}

static int matchModel(Mixer& m) {
	const int MAXLEN = 128;
	static Array<int> t(MEM);
	static int h = 0, ptr = 0, len = 0, result = 0, posnl = 0;
	if (!bpos) {
		h = (h * 997 * 8 + buf(1) + 1) & (t.size() - 1);
		if (len) {
			++len, ++ptr;
		} else {
			ptr = t[h];
			if (ptr && pos - ptr < buf.size()) {
				while (buf(len + 1) == buf[ptr - len - 1] && len < MAXLEN) ++len;
			}
		}
		t[h] = pos;
		result = len;
	}
	if (len) {
		if (buf(1) == buf[ptr - 1] && c0 == (buf[ptr] + 256) >> (8 - bpos)) {
			if (len > MAXLEN) len = MAXLEN;
			if (buf[ptr] >> (7 - bpos) & 1) {
				m.add(ilog(len) << 2);
				m.add(min(len, 32) << 6);
			} else {
				m.add(-(ilog(len) << 2));
				m.add(-(min(len, 32) << 6));
			}
		} else {
			len=0;
			m.add(0);
			m.add(0);
		}
	} else {
		m.add(0);
		m.add(0);
	}
	return result;
}

void indirectModel(Mixer& m) {
  static ContextMap cm(MEM*8, 5);
  static U32 t1[256];
  static U16 t2[0x10000];

  if (!bpos) {
    U32 d=c4&0xffff, c=d&255;
    U32& r1=t1[d>>8];
    r1=r1<<8|c;
    U16& r2=t2[c4>>8&0xffff];
    r2=r2<<8|c;
    U32 t=c|t1[c]<<8;
    cm.set(t&0xffff);
    cm.set(t&0xffffff);
    cm.set(t);
    t=d|t2[d]<<16;
    cm.set(t&0xffffff);
    cm.set(t);
  }
  cm.mix(m);
}

static int predictNext() {
	static ContextMap cm(MEM * 32, 8);
	static Mixer mixer(73, 1160, 4);
	static APM a1(0x100), a2(0x10000), a3(0x10000);
 
	c0 += c0 + y;
	if (c0 >= 256) {
		buf[pos++] = c0;
		c4 = (c4 << 8) + c0 - 256;
		c0 = 1;
	}   
	bpos = (bpos + 1) & 7;  
	int c1 = c4 & 0xff, c2 = (c4 & 0xff00) >> 8;
 
	mixer.update();
	int ismatch = ilog(matchModel(mixer));
	indirectModel(mixer);
 
	if (bpos == 0) {
		U32 z0 = 1 > pos ? 0 : buf(1);
		U8 pd = clen*2 > pos ? 0 : buf(clen*2);
		U8 ps = plen == 0 || plen == clen ? buf(clen) : 0;
		// cm.set(0);
		cm.set(c1);
		cm.set(c4 & 0x0000ffff);
		cm.set(c4 & 0x00ffffff);
		cm.set(c4);
		cm.set(z0);
		// cm.set(p3);
		cm.set(ps);
		cm.set(pd);
		cm.set(first);
		// cm.set(ismatch);
		// cm.set(ps>>4);
		// cm.set(ps&15);
	}
	int o = cm.mix(mixer);
	mixer.set(c1 + 8, 264);
	// mixer.set(c0, 256);
	mixer.set(o + ((c1 > 32) << 4) + ((bpos == 0) << 5) + ((c1 == c2) << 6), 128);
	mixer.set(c2, 256);
	mixer.set(ismatch, 256);
		 
	int pr0 = mixer.p();
	return (a1.p(pr0, c0) * 5
				 + a2.p(pr0, c0 + (c1 << 8)) * 15
				 + a3.p(pr0, hash(bpos, c1, c2) & 0xffff) * 12
				 + 16) >> 5;
}

U8 stack[10], top;
class Encoder {
private:
	const bool mode;
	U32 x, x1, x2;
	int p;
	Dat *f;
	int code(int i = 0) {
		p += p < 2048;  
		U32 xmid = x1 +((x2 - x1) >> 12) * p + (((x2 - x1) & 0xfff) * p >> 12);
		if (!mode) y = x <= xmid; else y = i;
		y ? (x2 = xmid) : (x1 = xmid+1);
		p = predictNext();
		while (((x1 ^ x2) & 0xff000000) == 0) {
			if (mode) putc(x2 >> 24, f);
			x1 <<= 8;
			x2 = (x2 << 8) + 255;
			if (!mode) x = (x << 8) + (getc(f) & 255);
		}
		return y;
	}
public:
	Encoder(bool m, Dat *f): mode(m), x(0), x1(0), x2(0xffffffff), p(2048), f(f) {
		if (!mode) {
			for (int i = 0; i < 4; ++i) {
				x = (x << 8) + (getc(f) & 255);
			}
		}
	}
	void flush() {
		putc(x1 >> 24, f);
	}  
	void compress(int c) {
		for (int i = 7; i >= 0; --i) {
			code((c >> i) & 1);
		}
	}
	void compress(U32 x, int bs) {
		top=0;
		while (bs--) {stack[top++]=x&255; x>>=8;}
		while (top) compress(stack[--top]);
	}
	int decompress() {
		int c = 0;
		for (int i = 0; i < 8; ++i) {
			c += c + code();
		}
		return c;
	}
	U32 decompress(int bs) {
		U32 r = 0;
		while (bs--) r = (r << 8) + decompress();
		return r;
	}
};

const int boc_generic = 0xb5ee9c72;
U32 magic, data_size;
int rcnt, ccnt, acnt, ref_bs, off_bs;

struct Cell {
	U8 d1, d2;
	int refs_cnt;
	bool special;
	int len, lmask, typ, l;
	string s, suf;
	std::vector<int> vref;

	Cell() {refs_cnt=len=typ=l=0; d1=d2=0; special=false; s=suf=""; vref.clear();}
	void reset() {refs_cnt=len=typ=l=0; d1=d2=0; special=false; s=suf=""; vref.clear();}
	void parse_d1(U8 x) {
		d1 = x;
		refs_cnt = d1 & 7;
		special = (d1 & 8) != 0;
		lmask = d1 >> 5;
	}
	void parse_d2(U8 x) {d2 = x; len = (d2 >> 1) + (d2 & 1);}
};
std::vector<Cell> vcells;

void parse_min(Dat *d) {
	vcells.clear();
	magic = getc(d, 4);
	U8 byte = getc(d);
	ref_bs = byte & 7;
	off_bs = getc(d);
	ccnt = getc(d, ref_bs);
	rcnt = getc(d, ref_bs);
	acnt = getc(d, ref_bs);
	data_size = getc(d, off_bs);
	L(i, 0, rcnt) int idx = getc(d, ref_bs);
	
	L(i, 0, ccnt) {
		U8 d1 = getc(d);
		U8 d2 = getc(d);
		Cell c;
		c.parse_d1(d1); c.parse_d2(d2);
		L(j, 0, c.len) c.s.pb(getc(d));
		L(j, 0, c.refs_cnt) c.vref.eb(getc(d, ref_bs));
		vcells.eb(c);
	}
}

Dat* dump_ultra() {
	clen = plen = first = 0;
	Dat *t = new Dat{tbuf, 0, 0};
	Encoder e(true, t);
	e.compress(ref_bs);
	e.compress(off_bs);
	e.compress(ccnt, ref_bs);
	e.compress(data_size, off_bs);

	std::vector<std::tuple<int,int,int>> ns;
	L(i, 0, ccnt) {
		if (!vcells[i].special) {
			int mx_ref = 0;
			for (auto r : vcells[i].vref) mx_ref = max(mx_ref, r);
			ns.emplace_back(vcells[i].len, mx_ref, i);
		}
	}
	std::sort(ns.begin(), ns.end());
	L(i, 1, ns.size()) {
		int c = std::get<2>(ns[i]);
		int p = std::get<2>(ns[i-1]);
		if (vcells[c].s == vcells[p].s) vcells[c].d1 |= 16;
	}

	L(i, 0, ccnt) e.compress(vcells[i].d1);
	L(i, 0, ccnt) if (!vcells[i].special) e.compress(vcells[i].d2);
	L(i, 0, ccnt) for (auto id : vcells[i].vref) e.compress(id-i, ref_bs);

	for (auto [_, __, i] : ns) {
		auto &v = vcells[i];
		if (v.d1 & 16) continue;
		clen = v.len; dx = v.d1;
		L(j, 0, clen) {
			e.compress(v.s[j]);
			if (j==0) first = v.s[0];
		}
		plen = clen;
	}
	clen = plen = first = 0;

	string s;
	std::vector<U8> dep;
	L(i, 0, ccnt) {
		if (vcells[i].special) {
			U8 typ = vcells[i].s[0];
			if (typ == 1) {
				U8 l = vcells[i].s[1];
				e.compress(4+l);
				L(j, 0, 32*l) s.pb(vcells[i].s[2+j]);
				L(j, 0, l) {
					dep.eb(vcells[i].s[2+32*l+2*j]);
					dep.eb(vcells[i].s[2+32*l+2*j+1]);
				}
			} else if (typ == 2) {
				e.compress(2);
				L(j, 0, 32) s.pb(vcells[i].s[1+j]);
			} else if (typ == 3) {
				e.compress(3);
				L(j, 0, 32) s.pb(vcells[i].s[1+j]);
				dep.eb(vcells[i].s[1+32]);
				dep.eb(vcells[i].s[1+32+1]);
			} else if (typ == 4) {
				e.compress(4);
				L(j, 0, 32) s.pb(vcells[i].s[1+j]);
				L(j, 0, 32) s.pb(vcells[i].s[1+32+j]);
				L(j, 0, 4) {
					dep.eb(vcells[i].s[1+64+j]);
				}
			}
		}
	}
	for (auto c : dep) e.compress(c);
	e.flush();
	Dat *o = new Dat{obuf, 0, 0};
	auto ls = s.size();
	putc(ls, 3, o);
	for (auto c : s) putc(c, o);
	L(i, 0, t->r) putc(t->p[i], o);
	return o;
}

void parse_ultra(Dat *d) {
	clen = plen = first = 0;
	size_t ls = getc(d, 3);
	Dat *t = new Dat{d->p, 3+ls, d->r};

	Encoder e(false, t);
	magic=boc_generic; rcnt = 1; acnt = 0;
	ref_bs = e.decompress();
	off_bs = e.decompress();
	ccnt = e.decompress(ref_bs);
	data_size = e.decompress(off_bs);
	vcells.resize(ccnt);
	L(i, 0, ccnt) vcells[i].reset();

	L(i, 0, ccnt) {
		U8 d1 = e.decompress();
		vcells[i].parse_d1(d1);
	}
	L(i, 0, ccnt) {
		if (!vcells[i].special) {
			U8 d2 = e.decompress();
			vcells[i].parse_d2(d2);
		}
	}
	std::vector<std::tuple<int,int,int>> ns;
	L(i, 0, ccnt) {
		int mx_ref = 0;
		L(j, 0, vcells[i].refs_cnt) {
			vcells[i].vref.eb(i + e.decompress(ref_bs));
			mx_ref = max(mx_ref, vcells[i].vref.back());
		}
		if (!vcells[i].special) {
			ns.eb(vcells[i].len, mx_ref, i);
		}
	}
	std::sort(ns.begin(), ns.end());
	for (auto [_, __, i] : ns) {
		if (vcells[i].d1 & 16) continue;
		clen = vcells[i].len;
		L(j, 0, clen) {
			vcells[i].s.pb(e.decompress());
			if (j == 0) first = vcells[i].s[0];
		}
		plen = clen;
	}
	int pid=-1;
	for (auto [_, __, i] : ns) {
		if (vcells[i].d1 & 16) vcells[i].s = vcells[pid].s, vcells[i].d1 ^= 16;
		pid = i;
	}	
	clen = plen = first = 0;
	L(i, 0, ccnt) {
		if (vcells[i].special) {
			U8 t = e.decompress();
			if (t > 4) {
				vcells[i].typ = 1;
				vcells[i].l = t - 4;
				vcells[i].s.pb(vcells[i].typ);
				vcells[i].s.pb(vcells[i].l);
			} else {
				vcells[i].typ = t;
				vcells[i].l = t - 2;
				vcells[i].s.pb(vcells[i].typ);
			}
		}
	}
	L(i, 0, ccnt) {
		if (vcells[i].special) {
			L(j, 0, vcells[i].l) {
				int a = e.decompress();
				int b = e.decompress();
				vcells[i].suf.pb(a);
				vcells[i].suf.pb(b);
			}
		}
	}
	d->l = 3; d->r = 3+ls;

	L(i, 0, ccnt) {
		if (vcells[i].special) {
			if (vcells[i].typ == 2) {
				L(j, 0, 32) {
					vcells[i].s.pb(getc(d));
				}
			} else {
				L(j, 0, 32*vcells[i].l) {
					vcells[i].s.pb(getc(d));
				}
			}
			vcells[i].s += vcells[i].suf;
			vcells[i].len = vcells[i].s.size();
			vcells[i].d2 = vcells[i].len * 2;
		}
	}
}

void dump_min(Dat *o) {
	putc(magic, 4, o);
	putc(ref_bs, o);
	putc(off_bs, o);
	putc(ccnt, ref_bs, o);
	putc(rcnt, ref_bs, o);
	putc(acnt, ref_bs, o);
	putc(data_size, off_bs, o);
	L(i, 0, rcnt) putc(0, ref_bs, o);
	L(i, 0, ccnt) {
		putc(vcells[i].d1, o);
		putc(vcells[i].d2, o);
		L(j, 0, vcells[i].len) putc(vcells[i].s[j], o);
		for (auto r : vcells[i].vref) putc(r, ref_bs, o);
	}
}

void load(string b) {
	td::BufferSlice data(td::base64_decode(b).move_as_ok());
	Dat *d = new Dat{data.data(), 0, data.length()};
	parse_ultra(d);
}

std::ofstream fds("dump.txt");
int tot=0;
void pre(string fn) {
	std::ifstream fin(fn);
	string m, b;
	fin >> m >> b;
	fin.close();
	td::BufferSlice data(td::base64_decode(b).move_as_ok());
	td::Ref<vm::Cell> root = vm::std_boc_deserialize(data).move_as_ok();
	td::BufferSlice s = vm::std_boc_serialize(root, 0).move_as_ok();
	Dat *d = new Dat{s.data(), 0, s.length()};
	parse_min(d);
	Dat *t = dump_ultra();
	string res = td::base64_encode(td::BufferSlice{t->p, t->r});
	fds << res << '\n';
	size_t ol = b.length(), nl = res.length();
	double score = 2000.0 * ol / (ol + nl);
	tot += res.length();
	std::cerr << tot << '\n';
}
string b20="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";
string b17="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";

int main() {
	// pre("test/1-020.txt"); pre("test/1-017.txt"); 
	// return 0;
	load(b20);load(b17);
	std::ios::sync_with_stdio(false); std::cin.tie(nullptr); std::cout.tie(nullptr);
	string m, b;
	std::cin >> m >> b;
	td::BufferSlice data(td::base64_decode(b).move_as_ok());
	if (m == "compress") {
		td::Ref<vm::Cell> root = vm::std_boc_deserialize(data).move_as_ok();
		td::BufferSlice s = vm::std_boc_serialize(root, 0).move_as_ok();
		Dat *d = new Dat{s.data(), 0, s.length()};
		parse_min(d);
		Dat *t = dump_ultra();
		std::cout << td::base64_encode(td::BufferSlice{t->p, t->r});
	} else {
		Dat *d = new Dat{data.data(), 0, data.length()};
		parse_ultra(d);
		Dat *o = new Dat{tbuf, 0, 0};
		dump_min(o);
		td::BufferSlice s{tbuf, o->r};
		auto root = vm::std_boc_deserialize(s).move_as_ok();
		auto odata = vm::std_boc_serialize(root, 31).move_as_ok();
		std::cout << td::base64_encode(odata);
	}
}